import datetime
import glob
import os.path
import re
from pathlib import Path

import psutil  # this package must be installed manually - not part of the tlomodel requirements

# This script must be run in the output directory
output_directory = "."


def do_task_directory(task_dir):
    link = f"{task_dir}"
    basename = '/'.join(task_dir.split('/')[1:])
    exit_status_path = f"{task_dir}/exit_status.txt"
    if os.path.exists(exit_status_path):
        exit_status = open(f"{task_dir}/exit_status.txt", "r").read()
        exit_status = exit_status.strip()
        if exit_status.strip() in ("0", "99"):
            status = "OK" if exit_status == "0" else "WAITING"
            print(f"<li><a href='{link}' style='color: black'>{basename} - {status}</a></li>")
        else:
            print(f"<li style='color: red'><a href='{link}' style='color: red'>{basename} - ERR</a></li>")
    else:
        # task hasn't terminated - check the process!
        task_info = f"{task_dir}/task.txt"
        pid_exists = False
        if os.path.exists(task_info):
            for line in open(task_info, 'r'):
                line = line.strip().split(' ')
                if line[0].startswith("PID"):
                    pid = line[1]
                    pid_exists = psutil.pid_exists(int(pid))
        status = "RUNNING" if pid_exists else "NO PROCESS"
        color = "black" if pid_exists else "red"
        print(f"<li style='color: {color}'><a href='{link}' style='color: {color}'>{basename} - {status}</a></li>")


def do_commit_directory(commit_dir):
    if not os.path.isfile(f'{commit_dir}/stdout.txt'):
        print(f'<h2>{commit_dir}</h2>')
        print('WARNING: stdout.txt not found; cannot continue')
        if not os.path.isfile(f'{commit_dir}/task.txt'):
            print('<br />WARNING: task.txt file not found')
        return
    # get information about commit
    for line in open(f'{commit_dir}/stdout.txt'):
        if re.search("^HEAD is now at", line):
            commit_message = line.rstrip().split(" ")[4:]
            commit_header = ' '.join(commit_message)
            print(f"<h2>{commit_header}</h2>")

            commit_link = f"https://github.com/UCL/TLOmodel/commit/{commit_message[0]}"
            top_dir = commit_dir.split("/")[-1]
            print(f"{top_dir}: <a href='{commit_link}'>commit</a>")

            pr_number = re.match(r"\(#(\d+)\)", commit_message[-1])
            if pr_number is not None and pr_number.groups():
                pr_link = f"https://github.com/UCL/TLOmodel/pull/{pr_number.groups()[0]}"
                print(f" <a href='{pr_link}'>pr</a>")
            break

    print("<ul>")
    #  for task in sorted(glob.glob(f'{commit_dir}/[0-9]*')):
    for task in sorted(glob.glob(commit_dir + '/**/stderr.txt', recursive=True)):
        do_task_directory(str(Path(task).parent))
    print("</ul>")


def run():
    style = """
    body {
    font-family: Helvetica, Arial;
    }
    a, u {
  text-decoration: none;
}
    """
    print(f"""<html><head><title>TLOmodel runs</title><style>{style}</style></head><body>""")
    print("<h1>TLOmodel runs</h1>")
    generated_time = datetime.datetime.now().strftime('%Y-%b-%d %H:%M')
    next_time = (datetime.datetime.now() + datetime.timedelta(minutes=5)).strftime('%Y-%b-%d %H:%M')
    print(f"Generated {generated_time} (next {next_time})")
    # loop over all commits that have been run (all begin with 202*
    for commit in sorted(glob.glob(f'{output_directory}/202[0-9]*'), reverse=True):
        do_commit_directory(commit)
    print("</body></html>")


if __name__ == '__main__':
    run()
