from huggingface_hub import snapshot_download
import argparse
import os

# read input from command line and download from snapshot
# python download.py --repo_id <repo_id> --path <path>

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="")
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--is_mk", type=str, default="False")
    args = parser.parse_args()
    path = args.path
    # os.listdir(path):
    # os.makedirs(path)
    repo_id = args.repo_id
    path = repo_id.split("/")[1]
    print(f"path {path}")
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    snapshot_download(repo_id=args.repo_id, local_dir=path)
    # print(f"Path {path} alerady exsists!")


main()
