from rest_framework import generics
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.accounts.models import User

from . import services
from .models import Document
from .permissions import CanManageDocuments
from .serializers import DocumentChunkSerializer, DocumentSerializer, DocumentUploadSerializer


def get_documents_queryset(user):
    queryset = Document.objects.select_related("course", "uploaded_by")
    if user.role == User.Role.CS_ADMIN:
        return queryset
    if user.role == User.Role.CS_COORDINATOR:
        return queryset.filter(course__in=user.coordinated_courses.all())
    return queryset.filter(course=user.course)


class DocumentListView(generics.ListAPIView):
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return get_documents_queryset(self.request.user)


class DocumentUploadView(generics.CreateAPIView):
    serializer_class = DocumentUploadSerializer
    permission_classes = [CanManageDocuments]


class DocumentDetailView(generics.RetrieveDestroyAPIView):
    serializer_class = DocumentSerializer
    permission_classes = [CanManageDocuments]

    def get_queryset(self):
        return get_documents_queryset(self.request.user)


class DocumentChunksView(generics.ListAPIView):
    serializer_class = DocumentChunkSerializer
    permission_classes = [CanManageDocuments]

    def get_queryset(self):
        document = generics.get_object_or_404(
            get_documents_queryset(self.request.user), pk=self.kwargs["pk"]
        )
        self.check_object_permissions(self.request, document)
        return document.chunks.all()


class DocumentReprocessView(APIView):
    permission_classes = [CanManageDocuments]

    def post(self, request, pk):
        document = generics.get_object_or_404(get_documents_queryset(request.user), pk=pk)
        self.check_object_permissions(request, document)
        services.reprocess_document(document)
        document.refresh_from_db()
        return Response(DocumentSerializer(document, context={"request": request}).data)
