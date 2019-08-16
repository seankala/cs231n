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