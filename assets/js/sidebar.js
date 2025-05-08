document.addEventListener('DOMContentLoaded', function() {
    // Only apply on mobile devices
    if (window.innerWidth <= 800) {
      // Get all sidebar headers
      var sidebarHeaders = document.querySelectorAll('.sidebar h3');
      
      // Add click event listener to each header
      sidebarHeaders.forEach(function(header) {
        header.addEventListener('click', function() {
          // Toggle active class on the parent sidebar
          this.parentNode.classList.toggle('active');
        });
      });
    }
  });