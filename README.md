# Who will win your fantasy football league?

If your fantasy football league does not automatically give you playoff odds (like my Yahoo facilitated league) then you may be wondering how screwed you are with your slow start to the season, or alternatively, how confident you should be betting on yourself in any intra-league side-wagers. Fortunately, by utilizing some basic machine learning principles we can get a good idea of each teams playoff chances with relative ease (given that we're willing to assume that past performance is generally predictive of future performance by team). The approach taken here will be to generate a posterior gaussian distribution of weekly point total for each individual team using a pooled prior distribution. The prior will be a gaussian distribution generated either by the previous season's weekly point totals for each team, or the current season's weekly point totals for each team. By using a conjugate prior analysis we can use a team's current season weekly point totals as a likelihood, update our priors and generate a fast predictive posterior distribution.

After getting each team's posterior distribution for points we can simulate an entire season by drawing each team's points from those posterior predictive gaussian distributions and then tallying the results of each game. We'll do this, seed the playoffs using each team's final record and point totals and then simulate each playoff game to get a simulated champion. By doing this hundreds or thousands of time, tallying the amount of times each team makes the playoffs and wins the championship, we can get a good idea of what each team's playoff and championship chances are.

Check out the notebook file "season_simulation.ipynb" to get an idea of how the scripts work as well as the math/theory behind the model. You can also see examples of some of the visuals produced. The functions in the module "season_sim_scripts.py" contain all the code needed to ingest, process, simulate, and plot results. 

You can easily visualize the results of the simulation to get a feel of how well each team is doing and how often they either make the playoffs or win the championship. 

Here we see a visual for each team's playoff chances according to the simulated model. 
![image](https://user-images.githubusercontent.com/79474788/142680363-e5050160-f47c-4b4d-bd84-240faddbc360.png)

Here we see a visual for each team's championship chances according to the simulated model. 
![image](https://user-images.githubusercontent.com/79474788/142680853-d3f344ab-1608-4740-a7ce-8c6df3a94362.png)

