"""Data schemas for function definitions, test inputs, and outputs."""

from typing import Any, Dict
from pydantic import BaseModel, Field


class ParameterProperty(BaseModel):
    """Specification of an individual function parameter."""

    type: str = Field(..., description="Data type of the parameter")


class ReturnProperty(BaseModel):
    """Specification of the function return value."""

    type: str = Field(..., description="Data type of the returned value")


class FunctionDefinition(BaseModel):
    """Complete specification of an available tool/function."""

    name: str = Field(..., description="Unique name of the function")
    description: str = Field(..., description="Summary of the function")
    parameters: Dict[str, ParameterProperty] = Field(
        default_factory=dict,
        description="Mapping of argument names to parameter specs",
    )
    returns: ReturnProperty = Field(
        ...,
        description="Return type specification",
    )


class PromptItem(BaseModel):
    """Input prompt format from the test dataset."""

    prompt: str = Field(..., description="Natural language user prompt")


class FunctionCallOutput(BaseModel):
    """Standardized output structure for a predicted function call."""

    prompt: str = Field(..., description="Original input prompt")
    name: str = Field(..., description="Selected function name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted arguments with validated values",
    )
