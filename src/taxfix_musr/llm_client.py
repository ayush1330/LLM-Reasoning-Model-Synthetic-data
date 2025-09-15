"""
LLM client wrapper for provider-agnostic chat interactions.

Provides a unified interface for different LLM providers with support
for structured output and error handling.
"""

import os
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv


class ModelConfig:
    """Configuration class for different OpenAI models."""
    
    CONFIGS = {
        "o3": {
            "supports_temperature": False,
            "supports_seed": False,
            "default_temperature": 1.0,
            "fallback_model": "gpt-4o-mini",
            "description": "Latest reasoning model (temperature fixed at 1.0)"
        },
        "o3-mini": {
            "supports_temperature": False,
            "supports_seed": False,
            "default_temperature": 1.0,
            "fallback_model": "gpt-4o-mini",
            "description": "Smaller reasoning model (temperature fixed at 1.0)"
        },
        "gpt-4o": {
            "supports_temperature": True,
            "supports_seed": True,
            "default_temperature": 0.2,
            "fallback_model": "gpt-4o-mini",
            "description": "GPT-4 Omni model with full parameter support"
        },
        "gpt-4o-mini": {
            "supports_temperature": True,
            "supports_seed": True,
            "default_temperature": 0.2,
            "fallback_model": "gpt-3.5-turbo",
            "description": "Smaller GPT-4 Omni model with full parameter support"
        },
        "gpt-4-turbo": {
            "supports_temperature": True,
            "supports_seed": True,
            "default_temperature": 0.2,
            "fallback_model": "gpt-4o-mini",
            "description": "GPT-4 Turbo model with full parameter support"
        }
    }
    
    @classmethod
    def get_config(cls, model: str) -> Dict:
        """Get configuration for a model."""
        return cls.CONFIGS.get(model, {
            "supports_temperature": True,
            "supports_seed": True,
            "default_temperature": 0.2,
            "fallback_model": "gpt-4o-mini",
            "description": "Unknown model, using default config"
        })
    
    @classmethod
    def get_optimal_params(cls, model: str, requested_temperature: float, requested_seed: int) -> Tuple[Dict, List[str]]:
        """Get optimal parameters for a model and return warnings."""
        config = cls.get_config(model)
        params = {"model": model}
        warnings = []
        
        # Handle temperature
        if config["supports_temperature"]:
            params["temperature"] = requested_temperature
        else:
            if requested_temperature != config["default_temperature"]:
                warnings.append(f"Model {model} doesn't support custom temperature, using default {config['default_temperature']}")
            # Don't set temperature parameter for models that don't support it
        
        # Handle seed
        if config["supports_seed"]:
            params["seed"] = requested_seed
        else:
            if requested_seed is not None:
                warnings.append(f"Model {model} doesn't support seed parameter")
        
        return params, warnings


class LLMClient:
    """LLM client for OpenAI API with smart model configuration."""

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gpt-4o-mini",  # Changed default to more reliable model
        temperature: float = 0.2,
        seed: int = 7
    ):
        """
        Initialize OpenAI client with smart model configuration.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use (defaults to gpt-4o-mini)
            temperature: Sampling temperature (defaults to 0.2)
            seed: Random seed for reproducibility (defaults to 7)
        """
        # Load environment variables
        load_dotenv()

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Store requested parameters
        self.requested_model = model
        self.requested_temperature = temperature
        self.requested_seed = seed
        
        # Set up model configuration
        self.model = model
        self.model_config = ModelConfig.get_config(model)
        
        # Get optimal parameters and warnings
        self.optimal_params, config_warnings = ModelConfig.get_optimal_params(
            model, temperature, seed
        )
        
        # Display configuration warnings
        for warning in config_warnings:
            print(f"⚠️  {warning}")
        
        # Display model info
        print(f"🤖 Using model: {model} - {self.model_config['description']}")
        
        # Test model availability with fallback
        self._setup_model_with_fallback()

    def _setup_model_with_fallback(self):
        """Set up model with intelligent fallback handling."""
        try:
            # Test model availability with a minimal request
            test_response = self.client.chat.completions.create(
                **self.optimal_params,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            print(f"✅ Model {self.model} is available and working")
            
        except Exception as e:
            print(f"❌ Model {self.model} failed: {e}")
            
            # Try fallback model
            fallback_model = self.model_config.get('fallback_model', 'gpt-4o-mini')
            print(f"🔄 Trying fallback model: {fallback_model}")
            
            self.model = fallback_model
            self.model_config = ModelConfig.get_config(fallback_model)
            self.optimal_params, fallback_warnings = ModelConfig.get_optimal_params(
                fallback_model, self.requested_temperature, self.requested_seed
            )
            
            for warning in fallback_warnings:
                print(f"⚠️  {warning}")
                
            try:
                # Test fallback model
                test_response = self.client.chat.completions.create(
                    **self.optimal_params,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1
                )
                print(f"✅ Fallback model {fallback_model} is working")
            except Exception as fallback_error:
                print(f"❌ Fallback model {fallback_model} also failed: {fallback_error}")
                raise RuntimeError(f"Both primary model {self.requested_model} and fallback {fallback_model} failed")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send chat messages to OpenAI and return response.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            OpenAI response text
        """
        try:
            # Use the pre-configured optimal parameters
            request_params = {
                **self.optimal_params,
                "messages": messages,
            }

            # Make the API call
            response = self.client.chat.completions.create(**request_params)

            # Extract and return the response text
            return response.choices[0].message.content

        except Exception as e:
            # Enhanced error reporting
            error_msg = f"LLM API call failed for model {self.model}: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model configuration."""
        return {
            "requested_model": self.requested_model,
            "actual_model": self.model,
            "requested_temperature": self.requested_temperature,
            "actual_params": self.optimal_params,
            "model_config": self.model_config,
            "supports_temperature": self.model_config["supports_temperature"],
            "supports_seed": self.model_config["supports_seed"]
        }


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: Optional[List[str]] = None):
        """
        Initialize mock client.

        Args:
            responses: List of predefined responses to return
        """
        self.responses = responses or []
        self.call_count = 0
        self.temperature = 0.2  # Default temperature

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Return mock response.

        Args:
            messages: List of message dicts (ignored)

        Returns:
            Mock response text
        """
        if not self.responses:
            # Return a valid, complete JSON response with all required fields
            return '''{
                "case_id": "mock_case",
                "narrative": "The taxpayer has a salary of $80,000 and made charitable donations of $5,000. The goal is to determine the taxable income after applying the donation cap rule.",
                "law_citations": [
                    {
                        "ref": "§DON-10pct",
                        "snippet": "Charitable contributions are deductible up to 10% of gross income."
                    }
                ],
                "reasoning_steps": [
                    {
                        "step": "Calculate gross income",
                        "claim": "Gross income is the sum of salary and freelance income",
                        "evidence": [
                            {
                                "source": "fact:salary",
                                "content": "80000"
                            }
                        ]
                    },
                    {
                        "step": "Apply donation cap",
                        "claim": "Donation deduction is limited to 10% of gross income",
                        "evidence": [
                            {
                                "source": "law:§DON-10pct",
                                "content": "Charitable contributions are deductible up to 10% of gross income."
                            }
                        ]
                    }
                ],
                "final_answer": {
                    "amount": 75000,
                    "explanation": "Taxable income calculated as gross income minus allowable deductions"
                }
            }'''

        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
