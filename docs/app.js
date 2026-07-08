/* Language + theme toggles, persisted in localStorage.
   Small, dependency-free, safe to load on every page. */
(function () {
  var root = document.documentElement;

  /* ---- language ---- */
  var savedLang = null;
  try { savedLang = localStorage.getItem("lang"); } catch (e) {}
  var lang = savedLang === "pt" ? "pt" : "en";
  setLang(lang);

  function setLang(l) {
    root.setAttribute("lang", l);
    root.setAttribute("data-lang", l);
    try { localStorage.setItem("lang", l); } catch (e) {}
  }

  var langBtn = document.getElementById("lang-toggle");
  if (langBtn) {
    langBtn.addEventListener("click", function () {
      setLang(root.getAttribute("data-lang") === "pt" ? "en" : "pt");
    });
  }

  /* ---- theme ---- */
  var savedTheme = null;
  try { savedTheme = localStorage.getItem("theme"); } catch (e) {}
  if (savedTheme === "light" || savedTheme === "dark") {
    root.setAttribute("data-theme", savedTheme);
  }

  var themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var current = root.getAttribute("data-theme");
      if (!current) {
        // no explicit theme yet: flip away from the OS preference
        var prefersDark = window.matchMedia &&
          window.matchMedia("(prefers-color-scheme: dark)").matches;
        current = prefersDark ? "dark" : "light";
      }
      var next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
    });
  }
})();
