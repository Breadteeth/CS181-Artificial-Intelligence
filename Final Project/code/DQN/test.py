import torch
from tankbattle.env.utils import Utils as GameUtils
from model.net import DeepQNetwork as NeuralNetwork
from tankbattle.env.engine import TankBattle as TankGame

# print("CUDA Available:", torch.cuda.is_available())
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

iterations_count = 10
exploration_rate = 0.05
# output_directory = "output/test/"
model_params_path = "parameter/predicted_0_0.3999961000195_951.pth"

def run_test(battle_env, dqn_model, initial_state, exploration, computation_device):
    accumulated_reward = 0.0
    current_state = initial_state
    # step_count = 0

    while True:
        battle_env.render()
        selected_action = dqn_model.select_action(current_state, exploration, computation_device)
        new_state, reward, is_done, _ = battle_env.step(selected_action)
        new_state = GameUtils.resize_image(new_state)

        # Optional: Save interaction data for analysis
        # interaction_data = ("current_state", selected_action, single_reward, "new_state", is_done)
        # GameUtils.store_data(interaction_data, f"{output_directory}interaction_{run_iteration}_{step_count}.txt")

        accumulated_reward += reward[0]

        if is_done:
            print(f"Total Score: {accumulated_reward}")
            break
        current_state = new_state
        # step_count += 1

if __name__ == "__main__":
    battle_simulation = TankGame(render=True, player1_human_control=False, player2_human_control=False, two_players=False,
                                speed=60, debug=False, frame_skip=5)

    for iteration in range(iterations_count):
        initial_game_state = battle_simulation.reset()
        game_state = GameUtils.resize_image(initial_game_state)
        base_dqn_model = NeuralNetwork(obs_shape=game_state.shape, action_count=battle_simulation.get_num_of_actions())
        trained_dqn_model = GameUtils.load_model(base_dqn_model, model_params_path)
        run_test(battle_simulation, trained_dqn_model, game_state, exploration_rate, compute_device)