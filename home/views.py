from rest_framework.views import APIView
from rest_framework.response import Response
from home.models import Location, UserQuery, DailyForecast, QueryResult ,VariableReading
from home.serializers import DailyForecastSerializer,LocationSerializer
from datetime import timedelta, datetime
from home.utility import analyze_climatology, DEFAULT_CONDITIONS,get_status
import requests
import os
import json
from rest_framework.permissions import AllowAny
import csv
from django.http import HttpResponse
from dotenv import load_dotenv

load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


    
# ----------------------------
# Forecast API
# ----------------------------
class ForecastAPI(APIView):
    def post(self, request):
        try:
            lat = float(request.data.get("lat"))
            lon = float(request.data.get("lon"))
            start_date_str = request.data.get("date")
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()

            # 1) Create Location & UserQuery
            location = Location.objects.create(lat=lat, lon=lon)
            query = UserQuery.objects.create(location=location, start_date=start_date)

            # 2) Loop 7 days to compute forecast
            forecasts = []
            for i in range(7):
                day = query.start_date + timedelta(days=i)
                result = analyze_climatology(
                    lat, lon, day.month, day.day,
                    start_year=2001, end_year=datetime.now().year,
                    window_days=7, conditions=DEFAULT_CONDITIONS
                )
                print(result["probabilities"])

                for r in result["probabilities"]:
                    # Use value if available, else 0
                    value = r.get("threshold_used", 0)
                    status = get_status(r["label"], value, r.get("prob", 0))
                    probability = r.get("prob")  # might be None
                    status = r.get("status") or "unknown"
                    unit = r.get("unit", "")
                    print(f"DEBUG: Unit value = '{unit}'") 

                    if unit == "°C HI":
                        title = "Heat Index"
                        variable_name = "HI"
                    elif unit == "°C":
                        title = "Temperature"
                        variable_name = "T2M"
                    elif unit == "m/s":
                        title = "Wind Speed" 
                        variable_name = "WS10M"
                    elif unit == "mm/day":
                        title = "Precipitation"
                        variable_name = "PRECTOTCORR"
                    else:
                        title = "Unknown"
                        variable_name = "UNKNOWN"
                    
                    df = DailyForecast.objects.create(
                        query=query,
                        date=day,
                        title=title,
                        value=float(value) if value is not None else 0.0,
                        status=r["label"],
                        probability=float(probability) if probability is not None else 0.0
                    )
                    forecasts.append(DailyForecastSerializer(df).data)
                    variable_reading, created = VariableReading.objects.get_or_create(
                        query=query,
                        variable=variable_name,  # Use unique variable name
                        date=day,
                        defaults={
                            'value': float(value) if value is not None else 0.0,
                            'unit': unit
                        }
                    )

            # 3) Send to Azure (replace YOUR_AZURE_URL & headers)
            try:
                azure_url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"

                azure_headers = {
                    "Content-Type": "application/json",
                    "api-key": AZURE_KEY
                }

                # Prepare messages for chat completion
                messages = [
                    {"role": "system", "content": "You are a weather summary assistant."},
                    {"role": "user", "content": f"Here are the 7-day forecasts:\n{json.dumps(forecasts)}\nPlease provide a JSON summary."}
                ]

                azure_payload = {
                    "messages": messages,
                    "max_tokens": 500
                }

                azure_response = requests.post(azure_url, headers=azure_headers, json=azure_payload, timeout=60)
                azure_response.raise_for_status()
                azure_summary = azure_response.json()

            except Exception as e:
                azure_summary = {"error": f"Azure request failed: {str(e)}"}

            # 4) Save Azure summary in QueryResult
            QueryResult.objects.create(
                query=query,
                raw_payload=azure_payload,
                summary=azure_summary
            )
            query_id = str(query.id)

            # 5) Return 7-day forecast + Azure summary
            return Response({
                "status": True,
                "message": "7-day forecast generated",
                "query_id": query_id, 
                "daily_forecasts": forecasts,
                "summary": azure_summary
            })

        except Exception as e:
            return Response({
                "status": False,
                "message": f"Error generating forecast: {str(e)}"
            })

    # Optional GET to retrieve existing forecasts by query ID
    def get(self, request):
            query_id = request.query_params.get("query_id")
            if not query_id:
                return Response({"status": False, "message": "query_id is required"})

            try:
                query = UserQuery.objects.get(id=query_id)
                forecasts = DailyForecastSerializer(query.daily_forecasts.all(), many=True).data
                summary = query.result.summary if hasattr(query, "result") else {}
                return Response({
                    "status": True,
                    "daily_forecasts": forecasts,
                    "summary": summary
                })
            except UserQuery.DoesNotExist:
                return Response({"status": False, "message": "Query not found"})
            
class DownloadWeatherCSV(APIView):
    permission_classes = [AllowAny]  # adjust if needed

    def get(self, request, q):
        query_id = request.query_params.get("query_id")
        if not query_id:
            return Response({"status": False, "message": "query_id is required"})

        try:
            query = UserQuery.objects.get(id=query_id)
            forecasts = query.daily_forecasts.all()
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename=forecast_{query_id}.csv'
            writer = csv.writer(response)
            writer.writerow(['date','title','value','status','probability'])
            for f in forecasts:
                writer.writerow([f.date, f.title, f.value, f.status, f.probability])
            return response
        except UserQuery.DoesNotExist:
                return Response({"status": False, "message": "Query not found"})
            