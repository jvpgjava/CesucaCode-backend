from django.urls import path

from .views import (
    ConversationDetailView,
    ConversationListCreateView,
    ConversationMessagesView,
    MessageFeedbackView,
    SendMessageView,
    SuggestionsView,
)

urlpatterns = [
    path("", ConversationListCreateView.as_view(), name="conversation-list-create"),
    path("suggestions/", SuggestionsView.as_view(), name="conversation-suggestions"),
    path("<int:pk>/", ConversationDetailView.as_view(), name="conversation-detail"),
    path("<int:pk>/messages/", ConversationMessagesView.as_view(), name="conversation-messages"),
    path(
        "<int:pk>/messages/<int:message_id>/feedback/",
        MessageFeedbackView.as_view(),
        name="conversation-message-feedback",
    ),
    path(
        "<int:pk>/messages/send/", SendMessageView.as_view(), name="conversation-send-message"
    ),
]
