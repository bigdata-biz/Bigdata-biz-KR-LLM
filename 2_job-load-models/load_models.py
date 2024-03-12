import subprocess

print(subprocess.run(["sh 2_job-load-models/load_models.sh"], shell=True))