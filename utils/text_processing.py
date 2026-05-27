"""Text preprocessing utilities used before transformer inference.

This module keeps deterministic input handling separate from Flask routes and
model code. The model layer receives bounded, normalized text while the API can
return clear validation errors to clients.
"""

import re
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Normalize and validate review text for moderation analysis."""
    
    @staticmethod
    def clean_text(text):
        """Remove noisy tokens and normalize whitespace before inference."""
        if not text:
            return ""
        
        # Remove URLs and email addresses as they add noise and minimal semantic value
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace: collapse multiple spaces, tabs, newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def validate_text(text, min_length=5, max_length=10000):
        """Return a validation tuple (is_valid, error_message).
        
        Used directly by API error responses to ensure frontend/backend
        validation consistency. Both layers use the same limits.
        """
        if not text:
            return False, "Text cannot be empty."
        
        # Strip whitespace for length validation
        text = text.strip()
        text_length = len(text)
        
        if text_length < min_length:
            return False, f"Text must be at least {min_length} characters (you have {text_length})."
        
        if text_length > max_length:
            return False, f"Text exceeds the {max_length:,} character safety limit (you have {text_length:,})."
        
        return True, ""
    
    @staticmethod
    def extract_keywords(text, num_keywords=5):
        """Extract lightweight keyword candidates using word frequency.

        This helper is not part of the prediction path today, but it is useful
        for future audit logs, report summaries, or reviewer-facing metadata.
        """
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Stop-word filtering keeps common grammar terms out of summaries.
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'with', 'about'
        }
        
        filtered_words = [w for w in words if w not in stop_words]
        
        freq = {}
        for word in filtered_words:
            freq[word] = freq.get(word, 0) + 1
        
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:num_keywords]]

def get_text_processor():
    """Return a new text processor instance."""
    return TextProcessor()

