import GA_RL_class
import rlcarsim
from Simulator import Environment,GUI,RL,Utils
args = Utils.parse_args()
args.control=='mvedql'
rlcarsim.rl_control_mvedql(config_file=args.config,arena_select=args.arena,load_weights=args.load_weights,testing=args.test)
