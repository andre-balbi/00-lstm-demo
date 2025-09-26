# Todo List for Visualization Update

## Objective
Update the `_plot_multistep_predictions` method in `src/utils/visualization.py` to handle cases where the horizon exceeds 12 by creating individual plots.

## Steps to Accomplish the Task

1. **Modify the layout logic**:
   - Update the existing layout logic to check if the `horizon` is greater than 12.

2. **Implement individual plots**:
   - If the `horizon` exceeds 12, create a loop to generate individual plots for each horizon value.

3. **Ensure existing functionality**:
   - Maintain the existing functionality for horizons up to 12.

## Todo List
- [ ] Update the layout logic in `_plot_multistep_predictions` to handle horizons greater than 12.
- [ ] Implement individual plots for each horizon when it exceeds 12.
- [ ] Test the changes to ensure that the plotting works correctly for all horizon values.