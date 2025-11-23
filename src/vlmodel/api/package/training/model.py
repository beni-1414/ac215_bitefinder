import torch
import torch.nn as nn
from transformers import CLIPModel, ViltModel, CLIPProcessor, ViltProcessor

'''
CLIP (Contrastive Language-Image Pre-training):
(*) Consists of two separate encoders: image encoder + text encoder (dual-encoder model)
(*) Uses contrastive learning to maximize similarity for matching image-text pairs and minimize similarity for non-matching image-text pairs
(+) Concatenate outputs from image encoder and text encoder together
(+) Add a classification head on top of dual-encoders

Initialization arguments:
- num_labels: number of classes
- pretrained: pretrained model name from Hugging Face model hub
- unfreeze_layers: number of layers to unfreeze from base VL model (last n layers)
- classifier_layers: number of layers in classification head
- dropout_prob: probability of element to be zeroed in dropout in classification head
- activation: activation function in classification head
'''


class CLIPForBugBiteClassification(nn.Module):
    def __init__(
        self,
        num_labels,
        pretrained='openai/clip-vit-base-patch32',
        unfreeze_layers=2,
        classifier_layers=1,
        dropout_prob=0.1,
        activation='relu',
    ):
        super().__init__()
        self.num_labels = num_labels
        self.pretrained = pretrained
        self.unfreeze_layers = unfreeze_layers
        self.classifier_layers = classifier_layers
        self.dropout_prob = dropout_prob
        self.activation = activation
        # Pre-trained CLIP model
        self.model = CLIPModel.from_pretrained(self.pretrained, use_safetensors=True, trust_remote_code=True)
        # Pre-trained CLIP processor
        self.processor = CLIPProcessor.from_pretrained(self.pretrained)
        # Linear classification head to project encoding space (image_embeds + text_embeds = projection_dim * 2) to label space
        vl_encoding_dim = 2 * self.model.config.projection_dim
        self.classifier = build_classifier(
            num_layers=classifier_layers,
            input_dim=vl_encoding_dim,
            output_dim=num_labels,
            dropout_prob=dropout_prob,
            activation_f=activation_funcs[activation],
        )

        # Unfreeze last N image + text encoder layers (and image + text projection layers)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if 'visual_projection' in name or 'text_projection' in name:
                param.requires_grad = True
        total_layers = 12
        last_layers = [total_layers - i - 1 for i in range(unfreeze_layers)]
        for name, param in self.model.named_parameters():
            if any(f'vision_model.encoder.layers.{i}' in name for i in last_layers):
                param.requires_grad = True
            if any(f'text_model.encoder.layers.{i}' in name for i in last_layers):
                param.requires_grad = True

        # Cross-entropy loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)  # CLIP
        output_embeds = torch.cat(
            [outputs.image_embeds, outputs.text_embeds], dim=-1
        )  # Concatenate image embeddings and text embeddings together as output
        logits = self.classifier(output_embeds)  # Pass output through classification head
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)  # Compute cross-entropy loss if training
        return {'loss': loss, 'logits': logits}


'''
ViLT (Vision-and-Language Transformer):
(*) Breaks images into patches and combines them with text tokens into a single sequence
(*) Processes image and text together using just one transformer (unified transformer architecture)
(+) Add a classification head on top of the [CLS] token

Initialization arguments:
- num_labels: number of classes
- pretrained: pre-trained model name from Hugging Face model hub
- unfreeze_layers: number of layers to unfreeze from base VL model (last n layers)
- classifier_layers: number of layers in classification head
- dropout_prob: probability of element to be zeroed in dropout in classification head
- activation: activation function in classification head
'''


class ViLTForBugBiteClassification(nn.Module):
    def __init__(
        self, num_labels, pretrained='dandelin/vilt-b32-mlm', unfreeze_layers=2, classifier_layers=1, dropout_prob=0.1, activation='relu'
    ):
        super().__init__()
        self.num_labels = num_labels
        self.pretrained = pretrained
        self.unfreeze_layers = unfreeze_layers
        self.classifier_layers = classifier_layers
        self.dropout_prob = dropout_prob
        self.activation = activation
        # Pre-trained ViLT model
        self.model = ViltModel.from_pretrained(self.pretrained, use_safetensors=True, trust_remote_code=True)
        # Pre-trained ViLT processor
        self.processor = ViltProcessor.from_pretrained(self.pretrained)
        # Linear classification head to project encoding space (of hidden_size dimensionality) to label space
        vl_encoding_dim = self.model.config.hidden_size
        self.classifier = build_classifier(
            num_layers=classifier_layers,
            input_dim=vl_encoding_dim,
            output_dim=num_labels,
            dropout_prob=dropout_prob,
            activation_f=activation_funcs[activation],
        )

        # Unfreeze last N layers
        total_layers = len(self.model.encoder.layer)
        last_layers = [total_layers - i - 1 for i in range(unfreeze_layers)]
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if any(f'encoder.layer.{i}' in name for i in last_layers):
                param.requires_grad = True

        # Cross-entropy loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, pixel_values, token_type_ids=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values
        )  # ViLT
        output_embeds = outputs.pooler_output  # Use [CLS] token embedding as output
        logits = self.classifier(output_embeds)  # Pass output through classification head
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)  # Compute cross-entropy loss if training
        return {'loss': loss, 'logits': logits}


'''
build_classifier: build classification head with multiple linear layers and RELU activations in between
'''


def build_classifier(
    num_layers: int,
    input_dim: int,
    output_dim: int,
    dropout_prob,
    activation_f,
):
    layers = []
    hidden_dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        layers.append(activation_f())
        hidden_dim = hidden_dim // 2
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


'''
model_classes: map of model name (in args and logs) to model class
'''
model_classes = {
    'clip': CLIPForBugBiteClassification,
    'vilt': ViLTForBugBiteClassification,
}

'''
activation_funcs: map of activation name (in args and logs) to activation function type
'''
activation_funcs = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
}
