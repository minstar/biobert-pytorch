import pdb
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from attacker import AdversarialAttack

class Hotflip(AdversarialAttack):
    """
    Attack adversarially with swap, insert, delete in Hotflip
    """
    def __init__(self, model, criterion, tokenizer, max_swaps=1, normalize_directions=True, show_progress=True, **kwargs):
        super().__init__(model, criterion, tokenizer, show_progress, **kwargs)
        self.max_swaps = max_swaps
        self.normalize_directions = normalize_directions

    def adversarial_attack(self, test_dataloader):
        instances_with_grads = []
        iterator             = tqdm(test_dataloader) if self.show_progress else test_dataloader

        for batch in iterator:
            self.batch_output = []
            self._vanilla_grads(batch)
            self._adversarial_tokens(batch)
            batch_output = self.update_output()
            instances_with_grads.extend(batch_output)

        return instances_with_grads

    def _register_forward_hook(self, ):
        """
        Register a forward hook on the embedding layer
        """

        def forward_hook(module, inputs, output):
            pass
        
        # Register the hook
        embedding_layer = self.get_embeddings_layer()
        handle          = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _vanilla_grads(self, batch):
        total_gradients = None
        handle = self._register_forward_hook()
        grads  = self._get_gradients(batch)
        handle.remove()

        if total_gradients is None:
            total_gradients = grads

        self.batch_output.append(total_gradients)

    def _adversarial_tokens(self, batch_output):
        """
        Generate adversarial token based on embedding matrix, gradient norm, and vocabulary.
        Most of the code is from https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py 
        """

        # get embedding matrix, gradient, and vocabulary
        embedding_layer = self.get_embeddings_layer()
        input_ids, outputs, grads = self.batch_output
        
        """
        input_ids : batch, seq length
        embedding_layer: vocab size, hidden dimension
        grads : batch, seq length, hidden dimension
        outputs : seq length, num labels
        """
        
        max_length = np.nonzero(input_ids.squeeze(0)).size()[0]
        max_swaps = min(self.max_swaps, max_length)
        
        embedding_matrix = embedding_layer.weight.detach()
        src_embeds = embedding_layer(input_ids).detach()
        
        new_embed_dot_grad = torch.einsum("bij,kj -> bik", (grads, embedding_matrix)) # batch, seq length, vocab size
        prev_embed_dot_grad = torch.einsum("bij,bij -> bi", (grads, src_embeds)) # batch, seq length

        # take the difference for each possible word. size is (batch, seq length, vocab size)
        neg_dir_dot_grad = prev_embed_dot_grad.unsqueeze(-1) - new_embed_dot_grad
        
        # multiply -1 to change the sign of the gradient for adversarial search
        pos_dir_dot_grad = -1 * neg_dir_dot_grad

        # renormalize if necessary
        if self.normalize_directions:
            # compute direction norm = distance word/substitution
            direction_norm = self._pairwise_distance(src_embeds, embedding_matrix)

            # renormalize
            pos_dir_dot_grad /= direction_norm
        
        # apply constraints
        pass
        
        score_at_each_step, best_at_each_step = pos_dir_dot_grad.max(2) # values and indices
        _, best_positions = score_at_each_step.topk(max_swaps)
        best_positions = torch.clamp(best_positions, 0, max_length-1)
        
        # create new adversarial examples
        adv_tokens = input_ids.clone()
        
        # assign new values
        index = best_positions.clone()
        src = best_at_each_step.gather(dim=1, index=best_positions.clone())
        
        adv_tokens.scatter_(dim=1, index=index, src=src)
        pdb.set_trace()
        self.batch_output.append(adv_tokens) # append adversarial token info.