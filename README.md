# Thesis

Data and Code for LLM evaluation, performance, and complexity analysis.

I am interested in examining the relationship between syntactic prompt complexity and LLM task and item level performance. In particular, I am interested in how this relationship changes with respect to model size and other experimental parameters. Various complexity measures are computed for each prompt using the [Stanza](https://github.com/stanfordnlp/stanza) parser to generate constituency and dependency grammar parses. I then test to see how well the different complexity scores corelate with the liklihood that each model will answer the task-item correctly.  
