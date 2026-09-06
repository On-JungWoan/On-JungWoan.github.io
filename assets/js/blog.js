/* Small enhancements; navigation and post lists also work without JavaScript. */
(function () {
  'use strict';

  document.querySelectorAll('[data-blog-collapse]').forEach(function (details) {
    var breakpoint = window.matchMedia(details.dataset.blogCollapse);
    function syncDisclosure() { details.open = !breakpoint.matches; }
    syncDisclosure();
    breakpoint.addEventListener('change', syncDisclosure);
  });

  // Let keyboard readers reach horizontally scrolling code and tables.
  document.querySelectorAll('.blog-prose pre, .blog-prose table').forEach(function (block) {
    block.tabIndex = 0;
  });
})();
