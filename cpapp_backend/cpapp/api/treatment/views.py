import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Q
from cpapp.models import Treatment

logger = logging.getLogger(__name__)


class TreatmentSearchView(APIView):
    """
    API view for searching treatments by name
    """
    
    def get(self, request):
        """
        Search treatments by name
        
        Query Parameters:
            q (str): Search query for treatment name
            limit (int, optional): Maximum number of results (default: 10)
            category (str, optional): Filter by category
            
        Returns:
            JSON response containing matching treatments
        """
        try:
            # Get search parameters
            search_query = request.GET.get('q', '').strip()
            limit = request.GET.get('limit', 10)
            category_filter = request.GET.get('category', '').strip()
            
            # Validate search query
            if not search_query:
                return Response({
                    "status": "error",
                    "message": "Search query 'q' parameter is required"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate limit parameter
            try:
                limit = int(limit)
                if limit <= 0 or limit > 100:
                    return Response({
                        "status": "error",
                        "message": "Limit must be between 1 and 100"
                    }, status=status.HTTP_400_BAD_REQUEST)
            except ValueError:
                return Response({
                    "status": "error",
                    "message": "Limit must be a valid integer"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Build query
            query = Q()
            
            # Add name search (case-insensitive)
            query &= Q(name__icontains=search_query)
            
            # Add category filter if provided
            if category_filter:
                query &= Q(category__icontains=category_filter)
            
            # Execute query
            treatments = Treatment.objects.filter(query).order_by('category', 'name')[:limit]
            
            # Serialize results
            results = []
            for treatment in treatments:
                results.append({
                    "id": treatment.id,
                    "name": treatment.name,
                    "category": treatment.category,
                    "created_at": treatment.created_at.isoformat() if treatment.created_at else None,
                    "updated_at": treatment.updated_at.isoformat() if treatment.updated_at else None
                })
            
            return Response({
                "status": "success",
                "data": {
                    "treatments": results,
                    "total_count": len(results),
                    "search_query": search_query,
                    "category_filter": category_filter if category_filter else None
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in treatment search: {e}")
            return Response({
                "status": "error",
                "message": "Internal server error occurred while searching treatments"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TreatmentCategoriesView(APIView):
    """
    API view for getting all available treatment categories
    """
    
    def get(self, request):
        """
        Get all available treatment categories
        
        Returns:
            JSON response containing list of categories
        """
        try:
            # Get distinct categories from treatments
            categories = Treatment.objects.values_list('category', flat=True).distinct().order_by('category')
            
            return Response({
                "status": "success",
                "data": {
                    "categories": list(categories),
                    "total_categories": len(categories)
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting treatment categories: {e}")
            return Response({
                "status": "error",
                "message": "Internal server error occurred while fetching categories"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR) 