import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow_addons.layers import CRF
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Target tags that need 90% F1 score
TARGET_TAGS = ['TEXT', 'TABLE', 'FORM']

class DocumentPreprocessor:
    """Process documents with special attention to spacing and structure."""
    
    @staticmethod
    def preprocess_document(doc_text: str) -> List[Dict]:
        """Process a document with special attention to spacing and structure."""
        lines = doc_text.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            # Keep original line to preserve spacing
            original_line = line
            line_stripped = line.strip()
            
            if not line_stripped and i > 0 and i < len(lines) - 1:
                # Track empty lines as they can be section separators
                processed_lines.append({
                    'text': '',
                    'is_empty': True,
                    'line_number': i
                })
                continue
            elif not line_stripped:
                continue
                
            # Spacing and indentation features
            leading_spaces = len(line) - len(line.lstrip())
            trailing_spaces = len(line) - len(line.rstrip())
            indent_level = leading_spaces // 2  # Assuming 2 spaces per indent level
            
            # Calculate space distribution (important for forms)
            internal_spaces = line.count(' ') - leading_spaces - trailing_spaces
            words = line_stripped.split()
            if len(words) > 1:
                avg_spaces_between_words = internal_spaces / (len(words) - 1)
            else:
                avg_spaces_between_words = 0
                
            # Check for form-like patterns
            has_form_separator = ':' in line_stripped or '=' in line_stripped
            label_value_match = re.search(r'^([A-Za-z0-9_\s]+)[:\s]+(.*)$', line_stripped)
            if label_value_match:
                potential_label = label_value_match.group(1).strip()
                potential_value = label_value_match.group(2).strip()
                is_likely_form_field = len(potential_label) < 30  # Most form labels are relatively short
            else:
                potential_label = ''
                potential_value = ''
                is_likely_form_field = False
                
            # Look for table-like structures
            spaces_between_chunks = re.findall(r'\S+(\s{2,})\S+', line)
            has_columnar_spacing = len(spaces_between_chunks) > 0
            consistent_spacing = len(set([len(s) for s in spaces_between_chunks])) <= 2 if spaces_between_chunks else False
            
            # Detect text-like content
            words = line_stripped.split()
            avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
            sentence_like = len(words) > 3 and avg_word_length > 3
            ends_with_punctuation = bool(re.search(r'[.!?]$', line_stripped))
            
            processed_lines.append({
                'text': original_line,  # Keep original to preserve exact spacing
                'text_stripped': line_stripped,
                'length': len(line),
                'stripped_length': len(line_stripped),
                'leading_spaces': leading_spaces,
                'trailing_spaces': trailing_spaces,
                'indent_level': indent_level,
                'internal_spaces': internal_spaces,
                'avg_spaces_between_words': avg_spaces_between_words,
                'has_form_separator': has_form_separator,
                'is_likely_form_field': is_likely_form_field,
                'potential_label': potential_label,
                'potential_value': potential_value,
                'has_columnar_spacing': has_columnar_spacing,
                'consistent_spacing': consistent_spacing,
                'words': len(words),
                'avg_word_length': avg_word_length,
                'sentence_like': sentence_like,
                'ends_with_punctuation': ends_with_punctuation,
                'has_numbers': bool(re.search(r'\d', line)),
                'is_capitalized': line_stripped.isupper(),
                'capital_ratio': sum(1 for c in line_stripped if c.isupper()) / max(len(line_stripped), 1),
                'position': i / len(lines),
                'line_number': i,
                'is_empty': False
            })
        
        return processed_lines

class BiLSTMCRFClassifier:
    """BiLSTM-CRF model for document structure classification."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        window_size: int = 10,
        lstm_units: Tuple[int, int] = (128, 64),
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize the BiLSTM-CRF classifier.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            window_size: Size of context window
            lstm_units: Number of units in LSTM layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Create model
        self.model, self.crf = self._create_model()
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Initialize preprocessor
        self.preprocessor = DocumentPreprocessor()
    
    def _create_model(self) -> Tuple[Model, CRF]:
        """Create the BiLSTM-CRF model architecture."""
        # Input layer
        input_layer = Input(shape=(None, self.input_dim))
        
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(LSTM(self.lstm_units[0], return_sequences=True))(input_layer)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        lstm2 = Bidirectional(LSTM(self.lstm_units[1], return_sequences=True))(lstm1)
        lstm2 = Dropout(self.dropout_rate)(lstm2)
        
        # Dense layer
        dense = TimeDistributed(Dense(self.num_classes))(lstm2)
        
        # CRF layer
        crf = CRF(self.num_classes)
        output = crf(dense)
        
        # Build and compile model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=crf.loss,
            metrics=[crf.accuracy]
        )
        
        return model, crf
    
    def _vectorize_features(self, features: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to numeric vectors."""
        # Select numeric features
        numeric_features = [
            'length', 'stripped_length', 'leading_spaces', 'trailing_spaces', 
            'indent_level', 'internal_spaces', 'avg_spaces_between_words',
            'has_form_separator', 'is_likely_form_field', 'has_columnar_spacing',
            'consistent_spacing', 'words', 'avg_word_length', 'sentence_like',
            'ends_with_punctuation', 'has_numbers', 'is_capitalized', 
            'capital_ratio', 'position'
        ]
        
        # Convert boolean features to int
        for f in features:
            for k in f:
                if isinstance(f[k], bool):
                    f[k] = int(f[k])
        
        # Extract vectors
        X = []
        for f in features:
            x = [f[k] for k in numeric_features if k in f]
            X.append(x)
        
        return np.array(X)
    
    def _prepare_sequence_data(
        self,
        documents: List[str],
        labels: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequential data with context windows."""
        all_sequences = []
        all_labels = []
        
        for i, doc in enumerate(documents):
            # Process document
            features = self.preprocessor.preprocess_document(doc)
            
            # Create feature vectors
            X = self._vectorize_features(features)
            
            # Split into windows with overlap
            for j in range(0, len(X) - self.window_size + 1, self.window_size // 2):
                window = X[j:j+self.window_size]
                all_sequences.append(window)
                
                # For labeled data, also collect labels
                if labels is not None:
                    window_labels = labels[i][j:j+self.window_size]
                    all_labels.append(window_labels)
        
        # Pad sequences to same length
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            all_sequences,
            padding='post',
            dtype='float32'
        )
        
        if labels is not None:
            padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
                all_labels,
                padding='post'
            )
            return padded_sequences, padded_labels
        
        return padded_sequences, None
    
    def _postprocess_predictions(
        self,
        line_predictions: np.ndarray,
        features: List[Dict]
    ) -> List[str]:
        """Apply logical post-processing to raw predictions."""
        tags = self.label_encoder.inverse_transform(line_predictions)
        
        # First pass: fix obvious inconsistencies
        for i in range(1, len(tags) - 1):
            # If a single line is differently classified than neighbors, likely an error
            if tags[i-1] == tags[i+1] and tags[i] != tags[i-1]:
                tags[i] = tags[i-1]
        
        # Second pass: merge isolated segments
        min_segment_length = 2
        current_tag = tags[0]
        segment_start = 0
        
        for i in range(1, len(tags)):
            if tags[i] != current_tag:
                # If segment is too short, merge with previous or next
                if i - segment_start < min_segment_length:
                    # Check which neighboring segment to merge with
                    if i < len(tags) - min_segment_length and tags[i] == tags[i+1]:
                        # Merge with next segment
                        tags[segment_start:i] = tags[i]
                    else:
                        # Merge with previous segment
                        tags[i] = current_tag
                
                current_tag = tags[i]
                segment_start = i
        
        # Third pass: apply form/table/text specific rules
        for i in range(len(tags)):
            # If line has form-like characteristics but isn't labeled as FORM
            if features[i]['is_likely_form_field'] and tags[i] != 'FORM':
                tags[i] = 'FORM'
                
            # If line has table-like characteristics but isn't labeled as TABLE
            if features[i]['has_columnar_spacing'] and features[i]['consistent_spacing'] and tags[i] != 'TABLE':
                tags[i] = 'TABLE'
                
            # If line is sentence-like but isn't labeled as TEXT
            if features[i]['sentence_like'] and features[i]['ends_with_punctuation'] and tags[i] != 'TEXT':
                tags[i] = 'TEXT'
        
        return tags
    
    def _convert_to_xml(
        self,
        original_doc: str,
        features: List[Dict],
        tags: List[str]
    ) -> str:
        """Convert line-level predictions to XML tagged format."""
        lines = original_doc.split('\n')
        result = []
        
        current_tag = None
        tag_content = []
        
        for i, (line, tag) in enumerate(zip(lines, tags)):
            if tag != current_tag:
                # Close previous tag if exists
                if current_tag is not None:
                    if tag_content:
                        result.append(f"<{current_tag}>")
                        result.extend(tag_content)
                        result.append(f"</{current_tag}>")
                    tag_content = []
                
                current_tag = tag
            
            # Add line to current tag content
            tag_content.append(line)
        
        # Add final tag
        if current_tag is not None and tag_content:
            result.append(f"<{current_tag}>")
            result.extend(tag_content)
            result.append(f"</{current_tag}>")
        
        return '\n'.join(result)
    
    def train(
        self,
        train_docs: List[str],
        train_labels: List[List[str]],
        val_docs: Optional[List[str]] = None,
        val_labels: Optional[List[List[str]]] = None,
        epochs: int = 15,
        batch_size: int = 32,
        early_stopping_patience: int = 3
    ) -> Dict:
        """
        Train the model on the provided documents.
        
        Args:
            train_docs: List of training documents
            train_labels: List of label sequences for training documents
            val_docs: Optional validation documents
            val_labels: Optional validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary containing training history
        """
        # Fit label encoder
        all_labels = [label for doc_labels in train_labels for label in doc_labels]
        self.label_encoder.fit(all_labels)
        
        # Encode training labels
        y_train = [self.label_encoder.transform(doc_labels) for doc_labels in train_labels]
        
        # Prepare training data
        X_train, y_train = self._prepare_sequence_data(train_docs, y_train)
        
        # Prepare validation data if provided
        if val_docs is not None and val_labels is not None:
            y_val = [self.label_encoder.transform(doc_labels) for doc_labels in val_labels]
            X_val, y_val = self._prepare_sequence_data(val_docs, y_val)
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=early_stopping_patience,
                    restore_best_weights=True
                )
            ]
        )
        
        return history.history
    
    def predict(self, documents: List[str]) -> List[str]:
        """
        Predict tags for the provided documents.
        
        Args:
            documents: List of documents to predict tags for
            
        Returns:
            List of XML-tagged documents
        """
        results = []
        
        for doc in documents:
            # Extract features
            doc_features = self.preprocessor.preprocess_document(doc)
            doc_X = self._vectorize_features(doc_features)
            
            # Create windows
            doc_X_seq = []
            for i in range(0, len(doc_X) - self.window_size + 1, self.window_size // 2):
                window = doc_X[i:i+self.window_size]
                if len(window) == self.window_size:
                    doc_X_seq.append(window)
            
            doc_X_seq = np.array(doc_X_seq)
            
            # Make predictions
            if len(doc_X_seq) > 0:
                window_preds = self.model.predict(doc_X_seq)
                
                # Convert from window predictions to line predictions
                line_preds = np.zeros(len(doc_features), dtype=int)
                counts = np.zeros(len(doc_features), dtype=int)
                
                for i, window_pred in enumerate(window_preds):
                    start_idx = i * (self.window_size // 2)
                    for j, pred in enumerate(window_pred):
                        if start_idx + j < len(line_preds):
                            line_preds[start_idx + j] += np.argmax(pred)
                            counts[start_idx + j] += 1
                
                # Average predictions where windows overlapped
                for i in range(len(line_preds)):
                    if counts[i] > 0:
                        line_preds[i] = line_preds[i] // counts[i]
                
                # Post-process predictions
                final_tags = self._postprocess_predictions(line_preds, doc_features)
                
                # Convert to XML format
                xml_output = self._convert_to_xml(doc, doc_features, final_tags)
                results.append(xml_output)
            else:
                results.append(doc)  # Return original if too short
        
        return results
    
    def save(self, model_dir: str):
        """Save the model and label encoder."""
        # Save model
        self.model.save(os.path.join(model_dir, "model"))
        
        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, os.path.join(model_dir, "label_encoder.joblib"))
    
    @classmethod
    def load(cls, model_dir: str) -> 'BiLSTMCRFClassifier':
        """Load a saved model and label encoder."""
        # Load model
        model = tf.keras.models.load_model(os.path.join(model_dir, "model"))
        
        # Load label encoder
        import joblib
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
        
        # Create classifier instance
        classifier = cls(
            input_dim=model.input_shape[-1],
            num_classes=len(label_encoder.classes_)
        )
        classifier.model = model
        classifier.label_encoder = label_encoder
        
        return classifier 