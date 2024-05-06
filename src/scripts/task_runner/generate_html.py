import datetime
import re
import sys
from pathlib import Path
from string import Template

max_commits = 50  # Number of runs to show in the generated html
page_title = "TLOmodel calibration analyses"


def main(output_directory: Path) -> None:
    """Generate and print the HTML page to stdout"""
    generated_time = datetime.datetime.now().strftime("%d %b %Y %H:%M")

    commit_dirs = [x for x in output_directory.iterdir() if x.is_dir()]
    commit_dirs = sorted(commit_dirs, reverse=True)
    commit_dirs = commit_dirs[:max_commits]
    body_html = "".join([get_html_for_commit(commit) for commit in commit_dirs])

    page_html = page_template.substitute(title=page_title,
                                         body=body_html,
                                         generated_time=generated_time)
    print(page_html)


def get_html_for_commit(commit_dir: Path) -> str:
    """Get the HTML for a single commit directory"""

    date = re.findall(r"\d{4}-\d{2}-\d{2}", commit_dir.name)[0]

    # some default values
    commit_title = commit_dir.name
    commit_links = list()
    commit_p_class = "incomplete"  # the commit is not shown by default

    # from the info.txt file, we know the message of the commit & can link to GitHub page
    if (commit_dir / "info.txt").is_file():
        with open(commit_dir / "info.txt") as info_file:
            commit_id, _, *commit_msg = info_file.readline().strip().split()
            gha_link = info_file.readline().strip()
            commit_msg = " ".join(commit_msg)
        commit_title = f"{date} - {commit_msg}"
        gh_link = f"https://github.com/UCL/TLOmodel/commit/{commit_id}"
        commit_links.append(f"<a href='{gh_link}'>commit</a>")
        commit_links.append(f"<a href='{gha_link}'>run</a>")

    # link to raw log files
    logs_dir = "021_long_run_all_diseases_run/0"
    if (commit_dir / logs_dir).is_dir():
        commit_links.append(f"<a href='./{commit_dir.stem}/{logs_dir}'>logs</a>")

    # if the post-process step has been completed, link to the results
    results_file = "022_long_run_all_diseases_process/index.html"
    if (commit_dir / results_file).is_file():
        commit_title = f"<a href='./{commit_dir.stem}/{results_file}'>{commit_title}</a>"
        commit_p_class = "completed"

    return commit_template.substitute(dir_title=commit_title,
                                      links=" ".join(commit_links),
                                      p_class=commit_p_class)


commit_template = Template("""<p class="$p_class">
    $dir_title
    $links<br>
</p>
""")

page_template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <title>$title</title>
    <style>
        body {
            font-family: Helvetica,Arial,sans-serif;
            font-weight: 400;
        }
        .incomplete {
            display: none;
        }
    </style>
    <script>
        // Toggle visibility of incomplete commits
        document.addEventListener("DOMContentLoaded", function () {
            let incompleteVisible = false;

            function toggleIncomplete() {
                document.querySelectorAll(".incomplete").forEach(incomplete => {
                    incomplete.style.display = incompleteVisible ? "none" : "block";
                });
                incompleteVisible = !incompleteVisible;
            }

            document.getElementById("toggleIncomplete").addEventListener("click", function (event) {
                event.preventDefault();
                toggleIncomplete();
            });
        });
    </script>
</head>
<body>
<h1>$title</h1>
<p style="font-size: small;">
    This page was generated on $generated_time. The 
    <a href="https://github.com/UCL/TLOmodel/actions/workflows/calibration.yaml">calibration workflow</a> runs every 
    night on the latest new commit on the master branch. <a href="#" id="toggleIncomplete">toggle incomplete</a>
</p>
$body
</body>
</html>
""")

if __name__ == "__main__":
    # Usage: python generate_html.py /path/to/output/directory
    main(Path(sys.argv[1]))
