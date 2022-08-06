import sys
sys.path.append("../")
import torch
import fire
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
os.environ["WANDB_SILENT"] = "True"
from lolbo.lolbo import LOLBOState
from lolbo.latent_space_objective import LatentSpaceObjective
try:
    import wandb
    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False


class Optimize(object):
    """
    Run LOLBO Optimization
    Args:
        task_id: String id for optimization task, by default the wandb project name will be f'optimize-{task_id}'
        seed: Random seed to be set. If None, no particular ranodm seed is set
        track_with_wandb: if True, run progreess will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid unsername necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"optimimze-{self.task_id}")
        minimize: If True we want to minimize the objective, otherwise we assume we want to maximize the objective
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_update_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        e2e_freq: Number of optimization steps before we update the models end to end (end to end update frequency)
        update_e2e: If True, we update the models end to end (we run LOLBO). If False, we never update end to end (we run TuRBO)
        reset_vae_at_tr_restart: If True, when we reset the trust region, also reset the vae to the initial weights (weights prior to any e2e updates)
        k: We keep track of and update end to end on the top k points found during optimization
        verbose: If True, we print out updates such as best score found, number of oracle calls made, etc. 
    """
    def __init__(
        self,
        task_id: str,
        seed: int=None,
        track_with_wandb: bool=False,
        wandb_entity: str="",
        wandb_project_name: str="",
        minimize: bool=False,
        max_n_oracle_calls: int=200_000,
        learning_rte: float=0.001,
        acq_func: str="ts",
        bsz: int=10,
        num_initialization_points: int=10_000,
        init_n_update_epochs: int=20,
        num_update_epochs: int=2,
        e2e_freq: int=10,
        update_e2e: bool=True,
        reset_vae_at_tr_restart: bool=False,
        k: int=1_000,
        verbose: bool=True,
        ):

        # add all local args to method args dict to be logged by wandb
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        self.seed = seed
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity 
        self.task_id = task_id
        self.max_n_oracle_calls = max_n_oracle_calls
        self.verbose = verbose
        self.num_initialization_points = num_initialization_points
        self.e2e_freq = e2e_freq
        self.update_e2e = update_e2e
        self.reset_vae_at_tr_restart = reset_vae_at_tr_restart
        self.set_seed()
        if wandb_project_name: # if project name specified
            self.wandb_project_name = wandb_project_name
        else: # otherwise use defualt
            self.wandb_project_name = f"optimimze-{self.task_id}"
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"

        # initialize latent space objective (self.objective) for particular task
        self.initialize_objective()
        assert isinstance(self.objective, LatentSpaceObjective), "self.objective must be an instance of LatentSpaceObjective"

        # initialize train data for particular task
        #   must define self.init_train_x, self.init_train_y, and self.init_train_z
        self.load_train_data()
        assert type(self.init_train_x) is list, "load_train_data() must set self.init_train_x to a list of xs"
        assert torch.is_tensor(self.init_train_y), "load_train_data() must set self.init_train_y to a tensor of ys"
        assert torch.is_tensor(self.init_train_z), "load_train_data() must set self.init_train_z to a tensor of zs"
        assert len(self.init_train_x) == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} xs, instead got {len(self.init_train_x)} xs"
        assert self.init_train_y.shape[0] == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} ys, instead got {self.init_train_y.shape[0]} ys"
        assert self.init_train_z.shape[0] == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} zs, instead got {self.init_train_z.shape[0]} zs"

        # initialize lolbo state
        self.lolbo_state = LOLBOState(
            objective=self.objective,
            train_x=self.init_train_x,
            train_y=self.init_train_y,
            train_z=self.init_train_z,
            minimize=minimize,
            k=k,
            num_update_epochs=num_update_epochs,
            init_n_epochs=init_n_update_epochs,
            learning_rte=learning_rte,
            bsz=bsz,
            acq_func=acq_func,
            verbose=verbose
        )


    def initialize_objective(self):
        ''' Initialize Objective for specific task
            must define self.objective obejct
            '''
        return self


    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_y (a tensor of corresponding latnet space points)
        '''
        return self


    def set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed) 
        return self
    

    def create_wandb_tracker(self):
        if self.track_with_wandb:
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config={k: v for method_dict in self.method_args.values() for k, v in method_dict.items()},
            ) 
            self.wandb_run_name = wandb.run.name
        else:
            self.tracker = None 
            self.wandb_run_name = 'no-wandb-tracking'
        
        return self


    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb:
            dict_log = {
                "best_found":self.lolbo_state.best_score_seen,
                "n_oracle_calls":self.lolbo_state.objective.num_calls,
                "total_number_of_e2e_updates":self.lolbo_state.tot_num_e2e_updates,
                "best_input_seen":self.lolbo_state.best_x_seen 
            }
            dict_log[f"TR_length"] = self.lolbo_state.tr_state.length
            self.tracker.log(dict_log)

        return self


    def run_lolbo(self): 
        ''' Main optimization loop
        '''
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        #main optimization loop
        while self.lolbo_state.objective.num_calls < self.max_n_oracle_calls:
            self.log_data_to_wandb_on_each_loop()
            # update models end to end when we fail to make
            #   progress e2e_freq times in a row (e2e_freq=10 by default)
            if (self.lolbo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e:
                self.lolbo_state.update_models_e2e()
                self.lolbo_state.recenter()
            else: # otherwise, just update the surrogate model on data
                self.lolbo_state.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            self.lolbo_state.acquisition()
            if self.lolbo_state.tr_state.restart_triggered:
                self.lolbo_state.initialize_tr_state()
                if self.reset_vae_at_tr_restart:
                    self.lolbo_state.objective.initialize_vae()
            # if a new best has been found, print out new best input and score:
            if self.lolbo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.lolbo_state.new_best_found = False

        # if verbose, print final results
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        # log top k scores and xs in table
        self.log_topk_table_wandb()

        return self 


    def print_progress_update(self):
        ''' Important data printed each time a new
            best input is found, as well as at the end 
            of the optimization run
            (only used if self.verbose==True)
            More print statements can be added her as desired
        '''
        if self.track_with_wandb:
            print(f"Optimization Run: {self.wandb_project_name}, {wandb.run.name}")
        print(f"Best X Found: {self.lolbo_state.best_x_seen}")
        print(f"Best {self.objective.task_id} Score: {self.lolbo_state.best_score_seen}")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.lolbo_state.objective.num_calls}")

        return self


    def log_topk_table_wandb(self):
        ''' After optimization finishes, log
            top k inputs and scores found
            during optimization '''
        if self.track_with_wandb:
            cols = ["Top K Scores", "Top K Strings"]
            data_list = []
            for ix, score in enumerate(self.lolbo_state.top_k_scores):
                data_list.append([ score, str(self.lolbo_state.top_k_xs[ix]) ])
            top_k_table = wandb.Table(columns=cols, data=data_list)
            self.tracker.log({f"top_k_table": top_k_table})
            self.tracker.finish()

        return self


    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)

if __name__ == "__main__":
    fire.Fire(Optimize)
