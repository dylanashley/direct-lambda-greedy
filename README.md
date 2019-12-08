# direct-lambda-greedy

This project is a look into how the the direct variance estimation method from *Comparing Direct and Indirect Temporal-Difference Methods for Estimating the Variance of the Return* by Craig Sherstan, Dylan R. Ashley, Brendan Bennett, Kenny Young, Adam White, Martha White, and Richard S. Sutton, affects the performance of the lambda-greedy algorithm from *A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning* by Adam White and Martha White. The abstract of the resulting technical report reads:

> We experiment with using the {$\lambda$}-greedy algorithm to simplify the task of tuning the trace-decay parameter on a vast network of temporal difference learning sub-agents, a problem that currently limits the utility of such networks. We find that {$\lambda$}-greedy can achieve good performance on our network. We also extend {$\lambda$}-greedy by using a recent, robust method of estimating the variance of the return. We find that this new variant has more stability in the $\lambda$ values selected from timestep to timestep but at the cost of decreased performance.

Included in this repository is the following:

- a technical report on the topic mentioned above
- the LaTeX code used to compile the report
- the source code for a series of experiments elaborated on in the report
