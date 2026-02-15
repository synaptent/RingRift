import { useEffect } from 'react';

const BASE_TITLE = 'RingRift';
const DEFAULT_DESCRIPTION =
  'A multiplayer territory-control strategy game. Place rings, form lines, claim territory, and outplay opponents on square and hexagonal boards.';

/**
 * Sets the document title and optionally the meta description for the current page.
 * Resets both on unmount.
 */
export function useDocumentTitle(subtitle?: string, description?: string) {
  useEffect(() => {
    document.title = subtitle
      ? `${subtitle} - ${BASE_TITLE}`
      : `${BASE_TITLE} - Multiplayer Strategy Game`;

    const metaDesc = document.querySelector('meta[name="description"]');
    if (metaDesc && description) {
      metaDesc.setAttribute('content', description);
    }

    return () => {
      document.title = `${BASE_TITLE} - Multiplayer Strategy Game`;
      const metaDescCleanup = document.querySelector('meta[name="description"]');
      if (metaDescCleanup) {
        metaDescCleanup.setAttribute('content', DEFAULT_DESCRIPTION);
      }
    };
  }, [subtitle, description]);
}
