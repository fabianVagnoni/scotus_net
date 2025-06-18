"""
Progress Bar Utilities
====================

Provides a unified tqdm progress bar interface with graceful fallback
when tqdm is not installed. Used across all SCOTUS AI pipeline scripts.

Usage:
    from src.utils.progress import tqdm, HAS_TQDM
    
    # Simple progress bar
    for item in tqdm(items, desc="Processing"):
        process(item)
    
    # Progress bar with configuration
    with tqdm(total=100, desc="Processing", disable=quiet and not HAS_TQDM) as pbar:
        for i in range(100):
            pbar.update(1)
            pbar.set_description(f"Processing item {i}")
"""

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
    
    # Re-export the real tqdm
    tqdm = _tqdm
    
except ImportError:
    HAS_TQDM = False
    
    # Fallback tqdm class for when tqdm is not installed
    class tqdm:
        """
        Fallback progress bar implementation that provides the same interface
        as tqdm but with minimal console output.
        """
        
        def __init__(self, iterable=None, total=None, desc="", disable=False, **kwargs):
            self.iterable = iterable or []
            self.total = total or (len(self.iterable) if hasattr(self.iterable, '__len__') else None)
            self.desc = desc
            self.n = 0
            self.disable = disable
            
            # Show initial message only if not disabled
            if not disable and desc:
                print(f"🔄 {desc}")
        
        def update(self, n=1):
            """Update progress by n steps."""
            self.n += n
            
        def set_description(self, desc):
            """Update the progress bar description."""
            self.desc = desc
            
        def close(self):
            """Close the progress bar."""
            pass
            
        def __enter__(self):
            """Context manager entry."""
            return self
            
        def __exit__(self, *args):
            """Context manager exit."""
            self.close()
        
        def __iter__(self):
            """Iterator support for using tqdm(iterable)."""
            for item in self.iterable:
                yield item
                self.update(1)
        
        @staticmethod
        def write(text):
            """Write text without interfering with progress bar (fallback just prints)."""
            print(text)


def get_progress_bar(iterable=None, total=None, desc="", disable=False, quiet=False, **kwargs):
    """
    Get a configured progress bar with smart disable logic.
    
    Args:
        iterable: Iterable to wrap (optional)
        total: Total number of items (optional)
        desc: Description for the progress bar
        disable: Force disable the progress bar
        quiet: If True and tqdm not available, disable progress bar
        **kwargs: Additional arguments passed to tqdm
    
    Returns:
        tqdm instance (real or fallback)
    """
    # Smart disable logic: disable if explicitly requested OR if quiet mode and no real tqdm
    should_disable = disable or (quiet and not HAS_TQDM)
    
    return tqdm(iterable=iterable, total=total, desc=desc, disable=should_disable, **kwargs) 