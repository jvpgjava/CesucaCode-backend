from django.urls import path

from .views import (
    ConversationDetailView,
    ConversationListCreateView,
    ConversationMessagesView,
    SendMessageView,
)

urlpatterns = [
    path("", ConversationListCreateView.as_view(), name="conversation-list-create"),
    path("<int:pk>/", ConversationDetailView.as_view(), name="conversation-detail"),
    path("<int:pk>/messages/", ConversationMessagesView.as_view(), name="conversation-messages"),
    path(
        "<int:pk>/messages/send/", SendMessageView.as_view(), name="conversation-send-message"
    ),
]
