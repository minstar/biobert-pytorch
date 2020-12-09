import pdb
import pickle

from dataset import NewsDataset
from model import DistilBertForSequenceClassification

from smooth_gradient import SmoothGradient
from integrated_gradient import IntegratedGradient

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertTokenizer

# from IPython.display import display, HTML

def main():
    config = AutoConfig.from_pretrained("bert-base-cased", num_labels=93)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased",
                from_tf=bool(".ckpt" in "bert-base-cased"),
                config=config)

    criterion = nn.CrossEntropyLoss()

    batch_size = 1
    # if torch.cuda.is_available():
    #     model.load_state_dict(
    #         torch.load('bert-base-cased')
    #     )
    # else:
    #     model.load_state_dict(
    #         torch.load('bert-base-cased', map_location=torch.device('cpu'))
    #     )
        
    # with open('../label_encoder.sklrn', 'rb') as f:
    #     le = pickle.load(f)

    test_example = [
        ["Interpretation of HuggingFase's model decision"], 
        ["Transformer-based models have taken a leading role "
        "in NLP today."]
    ]

    test_dataset = NewsDataset(
        data_list=test_example,
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings, 
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    integrated_grad = IntegratedGradient(
        model, 
        criterion, 
        tokenizer, 
        show_progress=True,
        encoder="bert"
    )
    instances = integrated_grad.saliency_interpret(test_dataloader)
    # pdb.set_trace()
    # coloder_string = integrated_grad.colorize(instances[0])
    # display(HTML(coloder_string))


if __name__ == '__main__':
    main()