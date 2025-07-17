from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from .models import ChatLog
from .langgraph_bot import process_user_query 

class ChatbotView(APIView):
    def post(self, request):
        question = request.data.get("question")
        if not question:
            return Response({"error": "Missing 'question'"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Run the LangGraph chatbot
            result = process_user_query(question)

            # Save to DB for review
            log = ChatLog.objects.create(
                question=result["question"],
                answer=result,
                status="pending_review",
                reviewed=False
            )

            return Response({
                "message": "Answer generated. Awaiting human review.",
                "id": log.id,
                "status": log.status
            }, status=status.HTTP_202_ACCEPTED)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["POST"])
def approve_answer(request, pk):
    try:
        log = ChatLog.objects.get(id=pk)
        log.status = "approved"
        log.reviewed = True
        log.save()
        return Response({"message": "Answer approved."})
    except ChatLog.DoesNotExist:
        return Response({"error": "Chat entry not found."}, status=404)

@api_view(["GET"])
def get_approved_answer(request, pk):
    try:
        log = ChatLog.objects.get(id=pk)
        if log.status != "approved":
            return Response({"error": "Answer not approved yet."}, status=403)
        return Response(log.answer)
    except ChatLog.DoesNotExist:
        return Response({"error": "Chat entry not found."}, status=404)

