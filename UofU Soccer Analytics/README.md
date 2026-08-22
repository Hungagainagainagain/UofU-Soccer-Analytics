# Competitor analyzed
As a data analyst for the Utah Division 1 Women’s Soccer Team, I noticed that the coaching staff faced a tedious challenge: watching hours of competitor match footage to identify potential tactical patterns. Each match is lengthy and requires full attention, and when multiplied by 
the last three games of each opponent—and then by the three staff members assigned to the task—the amount of time lost is substantial. 
That’s valuable time that could instead be spent developing our own players.

While higher-tier subscriptions offer API access to match footage and data, they come at a cost of thousands of dollars—a luxury we cannot afford. 
However, our existing subscription already provides access to Wyscout XML files containing detailed match event data. By leveraging this resource, we can automate much of the analysis, saving time and extracting actionable insights without the additional expense.

The goal of this analysis is to:
  - Analyze opponent corner kick attack formations – Identifying the attacking setups used during corners allows us to position our defense more effectively and target key opposing players most likely to receive the ball.
  - Identify opponent goalkeeper distribution patterns – Understanding where an opposing goalkeeper typically distributes the ball enables us to anticipate their build-up play and set up counter-defensive strategies to disrupt possession early.

The data is pulled from Wyscout in XML format, which included most recent 5 games of the opposing team. 
