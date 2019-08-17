# Progress Log

### August 14th, 2019 (Wednesday)

1. Batch normalization.

### August 16th, 2019 (Friday)

1. Gonna try and finish at least Batch Normalization today.
2. When working on the Batch Normalization Alternative Backward portion, I feel like I would be able to simplify the entire "underway" portion of the graph... We don't use $\partial L / \partial eps$ anyway so maybe we can skip the entire thing and go immediately into $H2$.
  * Okay maybe not the _entire_ underway... Hmm... What gradients would I not need?
  * Maybe I thought about it too hard... Let's just take out all of the intermediate values.
  * Actually I was kind of right... I think that one of the points that they're making is to just calculate everything on paper rather than actually code it out. I'll take a try at it.
  * I don't really understand the implementation... I need to try reading through all of the blogs in detail.

### August 17th, 2019 (Saturday)

1. Try to write out the entire graph again and get the thing right.
  * Questions I currently have are:
    * 1. When deriving the derivative for $dH_2$, why do we multiply by $dH_4$ and not $dH_3$? It just doesn't make sense to me even when I look at the computational graph.
      * Draw the computational graph following a different method, and try to derive it from there.
      * Figured out the problem. When I was deriving $\partial H_8 / \partial H_7$, I kept on writing the result as $\hat{x}$ rather than $\gamma$. This comes from my confusion of simple differentiation. $H_8 = \gamma \hat{x}$ and $H_7 = \hat{x}$, but I kept on getting the two confused. For example:
      $$
      \begin{align}
      &f(x) = 2x \\\\
      &\frac{df(x)}{dx} = 2
      \end{align}
      $$
      As you can see, I was differentiating the other way.