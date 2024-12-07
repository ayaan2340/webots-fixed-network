import torch
import os
from RecurrentNetwork import RecurrentNetwork


def save_model_portable(model, filepath):
    """
    Save model in a more portable format by only saving state dict
    """
    # Save only the state dictionary
    torch.save(model.state_dict(), filepath, _use_new_zipfile_serialization=True)


def load_model_cross_platform(model_class, filepath):
    """
    Load model in a cross-platform compatible way
    Args:
        model_class: The model class definition
        filepath: Path to the saved model file
    """
    try:
        # First try direct loading (might work if environments are compatible)
        model = torch.load(filepath)
        return model
    except (ModuleNotFoundError, RuntimeError) as e:
        print(f"Direct loading failed: {e}")
        try:
            # Initialize a new model instance
            model = model_class()

            # Try loading as state dict
            state_dict = torch.load(filepath, map_location=torch.device('cpu'))

            # If the file is a complete model rather than just the state dict,
            # try to extract the state dict
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()

            model.load_state_dict(state_dict)
            return model

        except Exception as e:
            print(f"State dict loading failed: {e}")
            try:
                # Last resort: try loading with pickle compatibility mode
                model = model_class()
                state_dict = torch.load(
                    filepath,
                    map_location=torch.device('cpu'),
                    pickle_module=pickle5.Pickle5
                )
                if hasattr(state_dict, 'state_dict'):
                    state_dict = state_dict.state_dict()
                model.load_state_dict(state_dict)
                return model
            except Exception as e:
                raise Exception(f"All loading attempts failed. Final error: {e}")


# When saving on Linux:
def save():
    model = RecurrentNetwork.load(
        "best_genomes/best_genome_gen_299.pt",
        input_size=6,
        hidden_size=10,
        output_size=2,
        genome_id=1)
    save_model_portable(model, 'best_model.pt')  # Note: using .pt extension instead of .h5

# When loading on Mac:


def load():
    try:
        model = load_model_cross_platform(RecurrentNetwork, 'best_model.pt')
    except Exception as e:
        print(f"Failed to load model: {e}")
