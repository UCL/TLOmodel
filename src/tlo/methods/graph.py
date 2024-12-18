

library(ggplot2)

# Plotting
ggplot(data, aes(x = value1, y = value2)) +
  geom_point(aes(color = r_incidence1549_6570_2_1_getp5)) +
  labs(title = "Scatter Plot of Data",
       x = "Value 1",
       y = "Value 2",
       color = "r_incidence1549_6570_2_1_getp5")
