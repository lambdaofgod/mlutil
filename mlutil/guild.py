import guild.ipy as guild
import guild.config as cfg
import glob
import os
import sys


def get_current_guild_env_path():
    return '/'.join(sys.executable.split('/')[:-2] + ['.guild']) 


def get_training_runs(operation_name, guild_env_path=get_current_guild_env_path()):
    """
    get guild runs with specified operation_name
    by default it will use guild from current environment
    """
    guild_home = cfg.SetGuildHome(guild_env_path)
    with guild_home:
        runs_df = guild.runs()
    training_runs_df = runs_df[runs_df['operation'] == operation_name]
    training_runs_df.loc[:, 'run'] = training_runs_df['run'][:].apply(str)
    return training_runs_df 


def get_weight_files(run_id):
    """
    get Keras model weight files for specified run_id
    """
    if run_id is None:
        return os.listdir()
    else:
        all_run_dirs = os.listdir('/'.join([current_env_guild, 'runs']))
        runs_df = get_training_runs(current_env_guild)
        run_path = [os.path.join(current_env_guild, 'runs', p) for p in all_run_dirs if p.startswith(run_id)][0]
        return sorted(glob.glob('/'.join([run_path, "*weight*"])))
