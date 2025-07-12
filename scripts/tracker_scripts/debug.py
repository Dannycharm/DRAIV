import os

output_path = '/scratch/dannycharm-alt-REU/DRAIV/scripts/tracker_scripts/'
output_dir = os.path.dirname(output_path)

if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Check if the directory is writable
if not os.access(output_dir, os.W_OK):
    raise PermissionError(f"Directory '{output_dir}' is not writable.")

