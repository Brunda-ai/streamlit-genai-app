import os

def save_uploaded_files(uploaded_files, prefix="temp_") -> list:
    """Saves raw files temporarily to disk and returns absolute local file paths."""
    saved_paths = []
    if not uploaded_files:
        return saved_paths

    for uploaded_file in uploaded_files:
        temp_path = f"{prefix}{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        saved_paths.append(temp_path)
    return saved_paths


def cleanup_local_paths(file_paths: list):
    """Safely cleans up temporary file system resources."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass 


def format_chat_history(session_history: list) -> list:
    """Transforms raw app logs into LangChain friendly history lists."""
    return [(msg["role"], msg["content"]) for msg in session_history]


def parse_source_metadata(document) -> tuple:
    """Extracts human-readable filenames and shifts page numbers to a 1-indexed baseline."""
    origin_file = os.path.basename(document.metadata.get('source', 'Unknown Manual'))
    page_offset = document.metadata.get('page', 0) + 1
    return origin_file, page_offset