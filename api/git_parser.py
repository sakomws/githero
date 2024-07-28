import os
import git
from pathlib import Path

def clone_repo(repo_url, clone_dir):
    try:
        # Clone the repository
        if not os.path.exists(clone_dir):
            os.makedirs(clone_dir)
        git.Repo.clone_from(repo_url, clone_dir)
        print(f'Repository cloned to {clone_dir}')
    except Exception as e:
        print(f'Error cloning repository: {e}')

def merge_python_files(clone_dir, output_file):
    try:
        with open(output_file, 'w') as outfile:
            file_count = 0
            for root, dirs, files in os.walk(clone_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as infile:
                            outfile.write(infile.read() + '\n')
                        file_count += 1
            if file_count == 0:
                print('No Python files found to merge.')
            else:
                print(f'All Python files have been merged into {output_file}')
    except Exception as e:
        print(f'Error merging Python files: {e}')
