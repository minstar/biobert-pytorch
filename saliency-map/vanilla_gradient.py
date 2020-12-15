import pdb
import torch

from tqdm import tqdm

from saliency_interpreter import SaliencyInterpreter

class VanillaGradient(SaliencyInterpreter):
    """
    Interprets the prediction using SmoothGrad (https://arxiv.org/abs/1312.6034)
    Registed as a 'SaliencyInterpreter' with name "vanilla-gradient"
    """
    def __init__(self, model, criterion, tokenizer, show_progress=True, **kwargs):
        super().__init__(model, criterion, tokenizer, show_progress, **kwargs)

    def saliency_interpret(self, test_dataloader):
        instances_with_grads = []
        iterator             = tqdm(test_dataloader) if self.show_progress else test_dataloader

        for batch in iterator:
            """
            We will store there batch outputs such as gradients, probs., tokens
            """
            self.batch_output = []
            self._vanilla_grads(batch)
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

