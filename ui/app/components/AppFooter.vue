<template>
  <footer class="app-footer">
    <div class="footer-inner">
      <p class="footer-brand">
        <span>© 2026 <a href="/" aria-label="Accueil Sawt AI">Sawt AI</a>.</span>
        <span>Tous droits réservés.</span>
      </p>

      <nav class="footer-links" aria-label="Liens utiles">
        <a href="/legal-notice">Mentions légales</a>
        <a href="/terms-of-service" aria-label="Conditions générales d’utilisation">CGU</a>
        <a href="/contact">Contact</a>
      </nav>
    </div>
  </footer>
</template>

<style scoped>
.app-footer {
  position: relative;
  z-index: 1;
  width: 100%;
  padding: 12px max(20px, env(safe-area-inset-right, 0px)) calc(16px + env(safe-area-inset-bottom, 0px)) max(20px, env(safe-area-inset-left, 0px));
}

.footer-inner {
  max-width: 980px;
  margin: 0 auto;
  padding: 8px 0 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px 24px;
}

.footer-brand {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 6px;
  margin: 0;
  font-size: 13px;
  line-height: 1.6;
  color: #8c9db4;
}

.footer-brand a {
  color: inherit;
  text-decoration: none;
}

.footer-links {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 0 18px;
}

.footer-links a {
  display: inline-flex;
  align-items: center;
  min-height: 44px;
  color: #a1b0c5;
  text-decoration: none;
  font-size: 12px;
  transition: color 0.2s ease, opacity 0.2s ease;
}

.footer-links a:hover {
  color: #fff;
}

.footer-links a:focus-visible,
.footer-brand a:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 4px;
}

@media (max-width: 640px) {
  .footer-inner {
    flex-direction: column;
    gap: 6px;
    text-align: center;
  }

  .footer-brand {
    justify-content: center;
  }

  .footer-links {
    width: 100%;
    justify-content: center;
    gap: 0 16px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .footer-links a {
    transition: none;
  }
}
</style>
