import numpy as np
from tqdm import tqdm

from saliency_interpreter import SaliencyInterpreter

class IntegratedGradient(SaliencyInterpreter):
    """
    Interprets the prediction using Integrated Gradients
    """
    def __init__(self, model, criterion, tokenizer, num_steps=20, show_progress=True, **kwargs):
        super().__init__(model, criterion, tokenizer, show_progress, **kwargs)
        self.num_steps = num_steps

    def saliency_interpret(self, test_dataloader):
        instances_with_grads = []
        iterator             = tqdm(test_dataloader) if self.show_progress else test_dataloader

        for batch in iterator:
            self.batch_output = []
            self._integrate_gradients(batch)
            batch_output = self.update_output()
            instances_with_grads.extend(batch_output)

        return instances_with_grads

    def _register_forward_hook(self, alpha, embeddings_list):
        """
        Register a forward hook on the embedding layer which scales the embeddings by alpha.
        """
        def forward_hook(module, inputs, output):
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach())

            # Scale the embedding by alpha
            output.mul_(alpha)
        
        embedding_layer = self.get_embeddings_layer()
        handle          = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _integrate_gradients(self, batch):

        ig_grads        = None
        embeddings_list = []

        for alpha in np.linspace(0, 1.0, num=self.num_steps, endpoint=False): # 0.05
            # Hook for modifying embedding value
            handle = self._register_forward_hook(alpha, embeddings_list)

            grads  = self._get_gradients(batch)
            handle.remove()

            if ig_grads is None:
                ig_grads = grads
            else:
                ig_grads = ig_grads + grads

        # Averaging of each gradient term
        ig_grads /= self.num_steps

        # Gradients get backward from network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        ig_grads *= embeddings_list[0]

        self.batch_output.append(ig_grads)

