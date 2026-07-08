import open_clip
import models

BIOMEDCLIP_TAG = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
BIOMEDCLIP_CONTEXT_LENGTH = 256
BIOMEDCLIP_EMBED_DIM = 512

if __name__ == "__main__":
    text = "A 45-year-old female presents with a lesion on the face. The patient is a smoker and has a history of skin cancer. The lesion has been itching, growing, hurting, changing, bleeding, raised."

    tokenizer = models.get_biomedclip_tokenizer()
    print(tokenizer(text, context_length=models.BIOMEDCLIP_CONTEXT_LENGTH))

