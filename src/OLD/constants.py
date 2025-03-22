"""Constants used across the document part classifier."""

# Primary tags that must be present in documents
PRIMARY_TAGS = {'TEXT', 'FORM', 'TABLE'}

# Special tokens for context handling
CONTEXT_TOKENS = {
    'sep': '[SEP]',  # Separator between context lines
    'line': '[LINE]'  # Marker for current line
}

# Feature extraction patterns
FORM_PATTERNS = {
    'field': r'[\_\.\:]{3,}',
    'label': r'(?i)(name|address|phone|email|date)[\s\:]+',
    'checkbox': r'^\s*\([X ]\)|\s*□'
}

TABLE_PATTERNS = {
    'columns': r'[\|\t]{2,}',
    'spacing': r'\s{3,}',
    'border': r'[+\-=]{3,}'
}

# Feature markers
FEATURE_MARKERS = {
    'field': '[HAS_FIELD]',
    'label': '[HAS_LABEL]',
    'checkbox': '[HAS_CHECKBOX]',
    'columns': '[HAS_COLUMNS]',
    'spacing': '[HAS_SPACING]',
    'border': '[HAS_BORDER]'
} 