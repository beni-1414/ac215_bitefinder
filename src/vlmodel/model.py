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
- model_name: pre-trained model name from Hugging Face model hub
- freeze_params: whether to freeze pre-trained model parameters during training
'''
class CLIPForBugBiteClassification(nn.Module):
    def __init__(self, num_labels, model_name='openai/clip-vit-base-patch32', freeze_params=True):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name) # Pre-trained CLIP model
        self.processor = CLIPProcessor.from_pretrained(model_name) # Pre-trained CLIP processor
        self.classifier = nn.Sequential( # Classification head (linear layer) to project image + text embeddings (image_embeds + text_embeds = projection_dim * 2) to label space
            nn.Dropout(0.1),
            nn.Linear(2 * self.model.config.projection_dim, num_labels)
        )
        # hidden_dim = self.model.config.projection_dim
        # self.classifier = nn.Sequential( # Classification head with two linear layers and GELU activation in between
        #     nn.Linear(2 * self.model.config.projection_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim, num_labels)
        # )
        if freeze_params: # Freeze pre-trained model parameters
            for name, param in self.model.named_parameters():
                if not any(x in name for x in ["classifier", "visual_projection", "text_projection"]): param.requires_grad = False
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values) # CLIP
        output_embeds = torch.cat([outputs.image_embeds, outputs.text_embeds], dim=-1) # Concatenate image embeddings and text embeddings together as output
        logits = self.classifier(output_embeds) # Pass output through classification head
        loss = None
        if labels is not None: loss = self.loss_fn(logits, labels) # Compute cross-entropy loss if training
        return {"loss": loss, "logits": logits}

'''
ViLT (Vision-and-Language Transformer):
(*) Breaks images into patches and combines them with text tokens into a single sequence
(*) Processes image and text together using just one transformer (unified transformer architecture)
(+) Add a classification head on top of the [CLS] token

Initialization arguments:
- num_labels: number of classes
- model_name: pre-trained model name from Hugging Face model hub
- freeze_params: whether to freeze pre-trained model parameters during training
- dropout_prob: dropout probability (0 means no dropout layer)
'''
class ViLTForBugBiteClassification(nn.Module):
    def __init__(self, num_labels, model_name='dandelin/vilt-b32-mlm', freeze_params=True):
        super().__init__()
        self.model = ViltModel.from_pretrained(model_name) # Pre-trained ViLT model
        self.processor = ViltProcessor.from_pretrained(model_name) # Pre-trained ViLT processor
        self.classifier = nn.Sequential( # Classification head (linear layer) to project encoding space (of hidden_size dimensionality) to label space
            nn.Dropout(0.1),
            nn.Linear(self.model.config.hidden_size, num_labels)
        )
        if freeze_params: # Freeze pre-trained model parameters
            for param in self.model.parameters(): param.requires_grad = False
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, pixel_values, token_type_ids=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values) # ViLT
        output_embeds = outputs.pooler_output # Use [CLS] token embedding as output
        logits = self.classifier(output_embeds) # Pass output through classification head
        loss = None
        if labels is not None: loss = self.loss_fn(logits, labels) # Compute cross-entropy loss if training
        return {"loss": loss, "logits": logits}