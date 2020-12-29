import pdb
import torch
import matplotlib
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.nn.functional import softmax

class AdversarialAttack:
    def __init__(self, model, criterion, tokenizer, show_progress=True, **kwargs):
        """
        param model: nn.Module object - can be HuggingFace's model or custom one.
        param criterion: nn.functional - torch criterion used to train your model. # need check
        param tokenizer: nn.Tokenizer - HuggingFace's tokenizer. # need check
        param show_progress: bool type - show tqdm progress bar. 
        param kwargs: encoder - string indicates the HuggingFace's encoder, that has 'embeddings' attribute.
                                Used if your model doesn't have 'get_input_embeddings' method to get access to encoder                                embeddings
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.show_progress = show_progress
        self.kwargs = kwargs

        self.batch_output = None

    def _get_gradients(self, batch):
        """
        set requires_grad to 'true' for all paramters, but save original values to resotre them later
        """
        embedding_gradients = []
        original_param_name_to_requires_grad_dict = {}
        
        for param_name, param in self.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True
        
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)
        loss  = self.forward_step(batch)

        self.model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        # restore the original requires_grad values of the parameters
        for param_name, param in self.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return embedding_gradients[0]

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the used to save the gradients of the embeddings for use in get_gradients()
        when there are multiple inputs (e.g., a passage and question), the hook will be called multiple times.
        We append all the embeddings gradients to a list.
        """

        def hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        embedding_layer = self.get_embeddings_layer()
        backward_hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return backward_hooks

    def get_embeddings_layer(self,):
        if hasattr(self.model, "get_input_embeddings"):
            embedding_layer = self.model.get_input_embeddings()
        else:
            encoder_attribute = self.kwargs.get("encoder")
            assert encoder_attribute, "Your model doesn't have 'get_input_embeddings' method, thus you " \
                    "have provide 'encoder' key argument while initializing SaliencyInterpreter object"
            embedding_layer = getattr(self.model, encoder_attribute).embeddings
        return embedding_layer

    @property
    def special_tokens(self, ):
        """
        some tokenizers don't have 'eos_token' and 'bos_token' attributes.
        Thus, we need some trick to get them.
        """

        if self.tokenizer.bos_token is None or self.tokenizer.eos_token is None:
            special_tokens     = self.tokenizer.build_inputs_with_special_tokens([])
            special_tokens_ids = self.tokenizer.convert_ids_to_tokens(special_tokens)
            self.tokenizer.bos_token, self.tokenizer.eos_token = special_tokens_ids

        special_tokens = self.tokenizer.eos_token, self.tokenizer.bos_token
        return special_tokens

    def _pairwise_dot_product(self, src_embeds, vocab_embeds, cosine=False):
        """
        Compute the cosine similarity between each word in the vocab and each word in the source
        """
        if cosine:
            src_embeds = F.normalize(src_embeds, dim=-1, p=2)
            vocab_embeds = F.normalize(vocab_embeds, dim=-1, p=2)
        # dot product
        dot_product = torch.einsum("bij,kj->bik", (src_embeds, vocab_embeds))
        return dot_product

    def _pairwise_distance(self, src_embeds, vocab_embeds, squared=False):
        """
        Compute the euclidean distance between each word in the vocab and each word in the source.
        """
        # compute square norm to avoid compute all the directions
        vocab_sq_norm = vocab_embeds.norm(p=2, dim=-1) ** 2
        src_sq_norm = src_embeds.norm(p=2, dim=-1) ** 2

        # dot product
        dot_product = self._pairwise_dot_product(src_embeds, vocab_embeds)
        
        # reshape for broadcasting
        vocab_sq_norm = vocab_sq_norm.unsqueeze(0).unsqueeze(0) # 1, 1, vocab size
        src_sq_norm = src_sq_norm.unsqueeze(2) # batch, seq length, 1

        # compute squared difference
        sq_norm = vocab_sq_norm + src_sq_norm - 2 * dot_product
        if squared:
            return sq_norm
        else:
            # relu + epsilon for numerical stability
            sq_norm = F.relu(sq_norm) + 1e-20
            
            # take the square root
            return sq_norm.sqrt()

    def forward_step(self, batch):
        """
        If your model receive inputs in another way or you computing not like in this example
        simply override this method.
        """
        input_ids      = torch.as_tensor(batch.input_ids).to(self.device).reshape((1, -1)) # batch.get('input_ids').to(self.device)
        attention_mask = torch.as_tensor(batch.attention_mask).to(self.device).reshape((1, -1)) # batch.get('attention_mask').to(self.device)
        outputs        = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

        _, _, num_label = outputs.shape
        """
        outputs : (batch, seq_length, feat_dim) => (seq_length, feat_dim)
        labels  : (batch, seq_length)           => (seq_length,)
        """
        outputs        = outputs.view(-1, num_label)
        labels         = torch.argmax(outputs, dim=1) # torch.argmax(outputs, dim=1)
        batch_losses   = self.criterion(outputs, labels)
        loss           = torch.mean(batch_losses) # mean average
        self.batch_output = [input_ids, outputs]
        return loss

    def update_output(self, ):
        """
        You can override this method if you want to change the format of outputs (e.g., storing gradients)
        """
        input_ids, outputs, grads, adv_tokens = self.batch_output

        probs         = softmax(outputs, dim=-1)
        probs, labels = torch.max(probs, dim=-1)

        tokens = [
            self.tokenizer.convert_ids_to_tokens(input_ids_)
            for input_ids_ in input_ids
        ]

        embedding_grads = grads.sum(dim=2)
        
        # norm for each sequence
        norms = torch.norm(embedding_grads, dim=1, p=2) # need check hyperparameter
        
        # normalizing
        for i, norm in enumerate(norms):
            embedding_grads[i] = torch.abs(embedding_grads[i]) / norm

        batch_output = []
        
        # check probs, labels shape
        labels   = torch.reshape(labels, (1, -1))
        probs    = torch.reshape(probs, (1, -1))
        iterator = zip(tokens, probs, embedding_grads, labels)

        for example_tokens, example_prob, example_grad, example_label in iterator:
            example_dict = dict()
            # as we do it by batches we has a padding so we need to remove it
            
            example_tokens = [t for t in example_tokens if t != self.tokenizer.pad_token]
            example_dict['tokens'] = example_tokens
            example_dict['grad']   = example_grad.cpu().tolist()[:len(example_tokens)]
            example_dict['label']  = example_label.cpu().tolist()[:len(example_tokens)] # example_label.item()
            example_dict['prob']   = example_prob.cpu().tolist()[:len(example_tokens)]  # example_prob.item() 

            batch_output.append(example_dict)

        return batch_output








