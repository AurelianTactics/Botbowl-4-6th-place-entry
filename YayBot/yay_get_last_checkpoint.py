#!/usr/bin/env python3
import os
from typing import List
import sys
from boto3 import client

def retrieve_checkpoint(path: str = "ray_results") -> str:
    """Returns a latest checkpoint unless there are none, then it returns None."""

    def all_dirs_under(path):
        """Iterates through all files that are under the given path."""
        for cur_path, dirnames, filenames in os.walk(path):
            for dir_ in dirnames:
                # print(dir_)
                yield os.path.join(cur_path, dir_)

    def retrieve_checkpoints(paths: List[str]) -> List[str]:
        checkpoints = list()
        for path in paths:
            for cur_path, dirnames, _ in os.walk(path):
                for dirname in dirnames:
                    if dirname.startswith("checkpoint_"):
                        checkpoints.append(os.path.join(cur_path, dirname))
                        # print("dirname ", print(os.path.getmtime(os.path.join(cur_path, dirname))))
                    # print(dirname)
        return checkpoints

    sorted_checkpoints = retrieve_checkpoints(
        sorted(
            filter(
                lambda x: x.startswith(path), all_dirs_under(path)
            ),
            key=os.path.getmtime
        )
    )[::-1]

    # above sort isn't working for me, going with last x digits
    ret_checkpoint_num = None
    ret_checkpoint = None
    for checkpoint in sorted_checkpoints:
        checkpoint_num = int(checkpoint.split("checkpoint_")[-1])
        if ret_checkpoint_num is None or checkpoint_num > ret_checkpoint_num:
            ret_checkpoint = checkpoint
            ret_checkpoint_num = checkpoint_num
            # print(ret_checkpoint)

    if ret_checkpoint is not None:
        return ret_checkpoint
    #     for checkpoint in sorted_checkpoints:
    #         if checkpoint is not None:
    #             return checkpoint
    # return None
    raise 1 == 0


# test_checkpoint = retrieve_checkpoint(
#     path="ray_results/botbowl_MA_IMPALANet_from_BC_then_Random-11-TDReward-IMPALANet")
# test_checkpoint

def retrieve_cloud_checkpoint(path_name):
    my_prefix = path_name #"ray-results/botbowl_MA_impalanet_sp-11-TDReward-IMPALANet/"
    checkpoint_str = "checkpoint-"
    #checkpoint_list = []
    last_checkpoint_path = None
    last_checkpoint_ts = None
    conn = client('s3')

    for key in conn.list_objects(Bucket='', Prefix=my_prefix)['Contents']:
        # print(key['Key'], key['LastModified'])
        if checkpoint_str in key['Key']:
            #checkpoint_list.append(key)
            ts = key['LastModified'].timestamp()  # checkpoint_list[0]['LastModified'].timestamp()
            if last_checkpoint_ts is None or ts > last_checkpoint_ts:
                #print("old path, new path", last_checkpoint_path, key['Key'])
                #print("old TS, new TS", last_checkpoint_ts, ts)
                last_checkpoint_path = key['Key']
                last_checkpoint_ts = ts

    return last_checkpoint_path

def main():
    path_name = sys.argv[1]
    # this gets checkpoint from local. doesn't actually get the last checkpoint, just the last dir of it
    # plus I'm getting hte highest number and not the last one modified
    # test_checkpoint = retrieve_checkpoint(
    #     path=path_name)
    checkpoint = retrieve_cloud_checkpoint(path_name)
    # example dir, split on last part
    #ray-results/botbowl_MA_impalanet_sp-11-TDReward-IMPALANet/PPO_my_botbowl_env_4cde9_00000_0_2022-04-19_00-18-57/checkpoint_001300/checkpoint-1300
    checkpoint_list = checkpoint.split("/")[:-1]
    checkpoint_dir = "/".join(checkpoint_list)
    checkpoint_file = checkpoint.split("/")[-1]
    if checkpoint is not None:
        #print(checkpoint)
        print('{} {} {}'.format(checkpoint, checkpoint_dir, checkpoint_file))
        #print('"{}" "{}" "{}"'.format(*my_python_function())
    else:
        # dummy directory that won't find checkpoint in. yes I should handle this more gracefully
        print("asdfasdfasdfasdfasdfasdfasdfasdfasdf")

    
if __name__ == "__main__":
    main()

