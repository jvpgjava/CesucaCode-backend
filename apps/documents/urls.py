from django.urls import path

from .views import (
    DocumentChunksView,
    DocumentDetailView,
    DocumentListView,
    DocumentReprocessView,
    DocumentUploadView,
)

urlpatterns = [
    path("", DocumentListView.as_view(), name="document-list"),
    path("upload/", DocumentUploadView.as_view(), name="document-upload"),
    path("<int:pk>/", DocumentDetailView.as_view(), name="document-detail"),
    path("<int:pk>/chunks/", DocumentChunksView.as_view(), name="document-chunks"),
    path("<int:pk>/reprocess/", DocumentReprocessView.as_view(), name="document-reprocess"),
]
