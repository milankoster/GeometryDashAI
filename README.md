# Temporary Notes
- base: normal
  -  Result: The model either constantly jumps or does nothing.
  - Theory: There's not enough fps, and the system gets stuck in mouse down mode. It may help to reduce click instead of pressing.
- base_input: click instead of press.
  - Result: The model never jumps.
  - Theory: Clicking may be an improvement, although it could cause issues later when flying. However, we don't reach that point as the model never jumps. This may be because it learns from frames the model is mid air.