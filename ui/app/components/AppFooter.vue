<template>
  <footer class="app-footer">
    <div class="footer-inner">
      <p class="footer-brand">© 2026 <a href="https://sawt-ai.netlify.app" target="_blank" rel="noreferrer"
          style="text-decoration: none;">Sawt AI</a> Tous
        droits réservés</p>

      <nav class="footer-links" aria-label="Liens utiles">
        <a href="https://github.com/nlabrazi/sawt-ai" target="_blank" rel="noreferrer">
          GitHub
        </a>
        <a href="https://portfolio-nlabrazi.netlify.app/" target="_blank" rel="noreferrer">
          Portfolio
        </a>
        <a href="mailto:na.labrazi@gmail.com">
          Contact
        </a>
      </nav>
    </div>
  </footer>
</template>

<style scoped>
.app-footer {
  position: relative;
  z-index: 1;
  padding: 0 16px 18px;
}

.footer-inner {
  max-width: 980px;
  margin: 0 auto;
  padding: 16px 18px;
  border-top: 1px solid rgba(148, 163, 184, 0.1);
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.footer-brand {
  margin: 0;
  font-size: 13px;
  color: #93c5fd;
}

.footer-links {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
}

.footer-links a {
  color: #cbd5e1;
  text-decoration: none;
  font-size: 14px;
  transition: color 0.2s ease, opacity 0.2s ease;
}

.footer-links a:hover {
  color: #fff;
}

@media (max-width: 768px) {
  .app-footer {
    padding: 0 14px 14px;
  }

  .footer-inner {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
}
</style>
