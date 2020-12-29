import pdb
import numpy as np
from tqdm import tqdm

from attacker import AdversarialAttack

class InputReduction(AdversarialAttack):
    """
    Attack adversarially with swap, insert, delete in Hotflip
    """
    def __init__(self, model, criterion, tokenizer, show_progress=True, **kwargs):
        super().__init__(model, criterion, tokenizer, show_progress, **kwargs)

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
        """

        # get embedding matrix, gradient, and vocabulary
        embedding_layer = self.get_embeddings_layer()
        input_ids, outputs, grads = self.batch_output
        pdb.set_trace()

        self.batch_output.append() # append adversarial toekn info.