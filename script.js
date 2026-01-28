const triggers = document.querySelectorAll('[data-accordion-trigger]');
const panels = document.querySelectorAll('[data-accordion-panel]');

triggers.forEach((trigger, index) => {
  trigger.addEventListener('click', () => {
    panels.forEach((panel, panelIndex) => {
      if (panelIndex === index) {
        panel.classList.toggle('active');
      } else {
        panel.classList.remove('active');
      }
    });
  });
});

document.querySelectorAll('[data-scroll]').forEach((button) => {
  button.addEventListener('click', () => {
    const target = document.querySelector(button.dataset.scroll);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
