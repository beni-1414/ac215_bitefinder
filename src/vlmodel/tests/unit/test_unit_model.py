import torch
import pytest
from unittest.mock import patch, MagicMock
from api.package.training.model import (
    CLIPForBugBiteClassification,
    ViLTForBugBiteClassification,
    build_classifier,
    activation_funcs,
)


@pytest.fixture
def fake_input():
    batch_size = 2
    seq_len = 5
    num_labels = 3
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_labels, (batch_size,))
    return input_ids, attention_mask, pixel_values, labels


@patch("api.package.training.model.CLIPModel.from_pretrained")
@patch("api.package.training.model.CLIPProcessor.from_pretrained")
def test_clip_init(mock_processor, mock_model):
    # Mock the model and processor
    mock_model.return_value = MagicMock(config=MagicMock(projection_dim=10), named_parameters=lambda: [])
    mock_processor.return_value = MagicMock()

    model = CLIPForBugBiteClassification(num_labels=3)
    assert isinstance(model.classifier, torch.nn.Module)
    assert model.processor is not None
    assert model.model is not None


@patch("api.package.training.model.CLIPModel.from_pretrained")
@patch("api.package.training.model.CLIPProcessor.from_pretrained")
def test_clip_forward(mock_processor, mock_model, fake_input):
    # Mock the outputs of the CLIP model
    mock_clip_output = MagicMock(image_embeds=torch.randn(2, 10), text_embeds=torch.randn(2, 10))
    mock_model.return_value = MagicMock(
        config=MagicMock(projection_dim=10),
        named_parameters=lambda: [],
        return_value=mock_clip_output,
        __call__=MagicMock(return_value=mock_clip_output),
    )
    mock_processor.return_value = MagicMock()

    input_ids, attention_mask, pixel_values, labels = fake_input
    model = CLIPForBugBiteClassification(num_labels=3)

    out = model(input_ids, attention_mask, pixel_values)
    assert 'logits' in out and 'loss' in out
    assert out['loss'] is None

    out = model(input_ids, attention_mask, pixel_values, labels=labels)
    assert out['loss'] is not None
    assert out['logits'].shape[0] == labels.shape[0]


@patch("api.package.training.model.ViltModel.from_pretrained")
@patch("api.package.training.model.ViltProcessor.from_pretrained")
def test_vilt_init(mock_processor, mock_model):
    # Mock the model and processor
    mock_model.return_value = MagicMock(config=MagicMock(hidden_size=10), encoder=MagicMock(layer=[0] * 12), named_parameters=lambda: [])
    mock_processor.return_value = MagicMock()

    model = ViLTForBugBiteClassification(num_labels=3)
    assert isinstance(model.classifier, torch.nn.Module)
    assert model.processor is not None
    assert model.model is not None


@patch("api.package.training.model.ViltModel.from_pretrained")
@patch("api.package.training.model.ViltProcessor.from_pretrained")
def test_vilt_forward(mock_processor, mock_model, fake_input):
    # Mock the outputs of the ViLT model
    mock_vilt_output = MagicMock(pooler_output=torch.randn(2, 10))
    mock_model.return_value = MagicMock(
        config=MagicMock(hidden_size=10),
        encoder=MagicMock(layer=[0] * 12),
        named_parameters=lambda: [],
        return_value=mock_vilt_output,
        __call__=MagicMock(return_value=mock_vilt_output),
    )
    mock_processor.return_value = MagicMock()

    input_ids, attention_mask, pixel_values, labels = fake_input
    model = ViLTForBugBiteClassification(num_labels=3)

    out = model(input_ids, attention_mask, pixel_values)
    assert 'logits' in out and 'loss' in out
    assert out['loss'] is None

    out = model(input_ids, attention_mask, pixel_values, labels=labels)
    assert out['loss'] is not None
    assert out['logits'].shape[0] == labels.shape[0]


def test_build_classifier_layers():
    classifier = build_classifier(
        num_layers=3,
        input_dim=16,
        output_dim=4,
        dropout_prob=0.1,
        activation_f=torch.nn.ReLU,
    )
    x = torch.randn(2, 16)
    out = classifier(x)
    assert out.shape[-1] == 4


def test_activation_funcs_mapping():
    assert activation_funcs['relu']() is not None
    assert activation_funcs['gelu']() is not None
