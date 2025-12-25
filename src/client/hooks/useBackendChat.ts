/**
 * @fileoverview useBackendChat Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for backend game chat.
 * It manages chat input state and submission, not game rules.
 *
 * This adapter:
 * - Tracks chat input text state
 * - Provides form submission handler
 * - Wraps the chat messages from useGameActions
 *
 * @see docs/architecture/BACKEND_GAME_HOST_DECOMPOSITION_PLAN.md
 */

import { useState, useCallback } from 'react';
import type { FormEvent } from 'react';

/**
 * Chat message structure.
 */
export interface ChatMessage {
  sender: string;
  text: string;
}

/**
 * Return type for useBackendChat hook.
 */
export interface UseBackendChatReturn {
  /** Current chat input value */
  chatInput: string;
  /** Set the chat input value */
  setChatInput: (input: string) => void;
  /** Chat messages from the backend */
  messages: ChatMessage[];
  /** Send a chat message */
  sendMessage: (text: string) => void;
  /** Handle form submission */
  handleSubmit: (e: FormEvent) => void;
}

/**
 * Custom hook for managing backend game chat state.
 *
 * Handles:
 * - Chat input text state
 * - Form submission with auto-clear
 * - Message sending wrapper
 *
 * Extracted from BackendGameHost to reduce component complexity.
 *
 * @param backendChatMessages - Messages from the backend (via useChatMessages)
 * @param sendChatMessage - Function to send messages (via useChatMessages)
 * @returns Object with chat state and actions
 */
export function useBackendChat(
  backendChatMessages: ChatMessage[],
  sendChatMessage: (text: string) => void
): UseBackendChatReturn {
  // Chat input state
  const [chatInput, setChatInput] = useState('');

  // Handle form submission
  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      if (!chatInput.trim()) return;

      sendChatMessage(chatInput);
      setChatInput('');
    },
    [chatInput, sendChatMessage]
  );

  // Send message wrapper (for direct calls without form)
  const sendMessage = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      sendChatMessage(text);
    },
    [sendChatMessage]
  );

  return {
    chatInput,
    setChatInput,
    messages: backendChatMessages,
    sendMessage,
    handleSubmit,
  };
}

export default useBackendChat;
