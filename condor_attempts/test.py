import htcondor
import argparse
import os

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

parser = argparse.ArgumentParser()
#parser.add_argument("--basedir", type=str, help="SPANet directory", default=os.getcwd())
parser.add_argument("--of", type=str, help="option file (relative to basedir)", required=False, default="base")
parser.add_argument("--epochs", type=str, help="number of epochs", required=True)
parser.add_argument("--gpus", type=str, help="number of gpus", required=True)
parser.add_argument("--test", action="store_true", help="Do no run condor job but interactively")
args = parser.parse_args()

# Checking the input files exists
#os.makedirs(args.basedir, exist_ok=True)
#os.makedirs(f"{args.basedir}/condor_logs", exist_ok=True)

#if not os.path.exists(os.path.join(args.basedir, "options_files",  args.of)):
    #raise ValueError(f"Option file does not exists: {args.of}")

if args.test:
    os.system(f"sh test.sh")
    exit(0)

sub = htcondor.Submit()
sub['Executable'] = "test.sh"
sub["arguments"] = f"{args.of} {args.epochs} {args.gpus}"
sub['Error'] = "/afs/cern.ch/user/y/ymaidann/condor_logs/error/training-$(ClusterId).$(ProcId).err"
sub['Output'] = "/afs/cern.ch/user/y/ymaidann/condor_logs/output/training-$(ClusterId).$(ProcId).out"
sub['Log'] = "/afs/cern.ch/user/y/ymaidann/condor_logs/log/training-$(ClusterId).log"
sub['MY.SendCredential'] = True
sub['+JobFlavour'] = "testmatch"
#sub["transfer_input_files"] = "/eos/user/y/ymaidann/eth_project/Spanet_project/SPANet/"
sub["when_to_transfer_output"] = "ON_EXIT"
sub['request_cpus'] = '4'
sub['request_gpus'] = '1'

schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    cluster_id = sub.queue(txn)
    print(cluster_id)
    # Saving the log
    with open(f"/afs/cern.ch/user/y/ymaidann/condor_logs/training_jobs.csv", "a") as l:
        l.write(f"{cluster_id};{args.of};{args.epochs};{args.gpus}\n")


