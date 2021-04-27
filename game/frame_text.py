
main_info_text = 'The target high score is the score in the game that the agent will try to achieve. ' \
                 'The agent will retry after each death until the score has been reached.\n' \
                 'The learning rate determines how much the agent will change its behavior after each evaluation ' \
                 'of its performance.' \
                 'Higher learning rates speed up learning, but cause greater changes to the agent on each ' \
                 'evaluation and can lead to overfitting and poor performance.' \
                 'Lower learning rates will cause the agent to make smaller adjustments and learn slowly, but may ' \
                 'encourage the agent to repeat the same pattern of suboptimal behaviors.'


plot_tab_text = 'Plots display the average location of game object inputs to the neural network for each output action taken. ' \
                'For each window of 20 decisions, the average inputs are plotted on the graph. ' \
                'Each plot corresponds to one of the four possible output decisions. Inputs are plotted as position relative to Pac-Man. ' \
                'Decisions are grouped by the input value of whether a power pellet was active and can be toggled on or off.'

about_tab_text = 'This application demonstrates and reports on the performance of an artificial intelligence agent which plays' \
                 'the classic arcade game, Pac-Man. The agent operates using a neural network structure, as well as Q-Learning ' \
                 'principles to help it evolve. The agent makes decisions by using locations of game objects as inputs to ' \
                 'the network and selects an action from the four movement directions in the game. The user can provide a ' \
                 'target high score and learning rate for the agent in order to see the effects on the agent\'s performance. ' \
                 'It serves as a familiar introduction to the application of artificial intelligence on real problems.'

credits_text = 'This project was made as a senior capstone at Oakland University by \nSydney Hill, James Lynott, ' \
               'Kyle McKinley, Dan Ruyle, and Michael Smith \n'

target_high_score_tooltip = 'Set the target high score for the agent to achieve'

learning_rate_tooltip = 'Set learning rate of the agent\nSee \"Information\" for details'

plot_active_tooltip = 'Show or hide decisions that were made while a power pellet was active'

plot_inactive_tooltip = 'Show or hide decisions that were made while a power pellet was not active'

plot_rand_tooltip = 'Show or hide decisions that were made at random'

visualizer_tooltip = 'Set visualizer tooltip'

starting_locations_tooltip = 'Choose whether to start Pac-Man in a central location on the map\nor a random location (default)'

start_button_tooltip = 'Start the agent'


