SHELL := /bin/bash
SRC_NAMESPACE ?= src
MONITORING_NAMESPACE ?= monitoring
INGRESS_NAMESPACE ?= ingress-nginx
LOGGING_NAMESPACE ?= logging
ENV_FILE ?= .env

# Load environment variables from .env if it exists
ifneq (,$(wildcard $(ENV_FILE)))
    include $(ENV_FILE)
    export
endif


help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

build-images: ## Build Backend & Frontend Docker images locally
	@echo "Building Docker images..."
	docker build -t vqa-backend:latest ./backend
	docker build -t vqa-frontend:latest ./frontend
	@echo "Images built successfully: vqa-backend:latest, vqa-frontend:latest"

create-namespaces: ## Create separated namespaces if they don't exist
	@echo "Creating namespaces ($(SRC_NAMESPACE), $(MONITORING_NAMESPACE), $(INGRESS_NAMESPACE), $(LOGGING_NAMESPACE))..."
	@kubectl create namespace $(SRC_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@kubectl create namespace $(MONITORING_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@kubectl create namespace $(INGRESS_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@kubectl create namespace $(LOGGING_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -

create-secrets: create-namespaces ## Create or update Kubernetes secrets in src namespace
	@echo "Creating / updating vqa-secrets in $(SRC_NAMESPACE)..."
	@kubectl create secret generic vqa-secrets \
		--namespace=$(SRC_NAMESPACE) \
		--from-literal=GROQ_API_KEY="$(GROQ_API_KEY)" \
		--from-literal=VLLM_API_BASE="$(VLLM_API_BASE)" \
		--from-literal=VLLM_API_KEY="$(VLLM_API_KEY)" \
		--from-literal=LANGFUSE_PUBLIC_KEY="$(LANGFUSE_PUBLIC_KEY)" \
		--from-literal=LANGFUSE_SECRET_KEY="$(LANGFUSE_SECRET_KEY)" \
		--from-literal=LANGFUSE_HOST="$(LANGFUSE_HOST)" \
		--from-literal=POSTGRES_USER="$(POSTGRES_USER)" \
		--from-literal=POSTGRES_PASSWORD="$(POSTGRES_PASSWORD)" \
		--from-literal=POSTGRES_DB="$(POSTGRES_DB)" \
		--from-literal=DATABASE_URL="$(DATABASE_URL)" \
		--from-literal=MINIO_ROOT_USER="$(MINIO_ROOT_USER)" \
		--from-literal=MINIO_ROOT_PASSWORD="$(MINIO_ROOT_PASSWORD)" \
		--from-literal=DISCORD_WEBHOOK_URL="$(DISCORD_WEBHOOK_URL)" \
		--from-literal=EMBEDDER_URL="$(EMBEDDER_URL)" \
		--from-literal=EMBEDDER_API_KEY="$(EMBEDDER_API_KEY)" \
		--dry-run=client -o yaml | kubectl apply -f -
	@echo "Secrets applied to $(SRC_NAMESPACE)."

deploy: create-secrets ## Deploy EVERYTHING to separated namespaces (ingress-nginx, src, monitoring, logging)
	@echo "============================================="
	@echo "1/5. Deploying NGINX Ingress Controller ($(INGRESS_NAMESPACE))..."
	@echo "============================================="
	-helm upgrade --install ingress-nginx ./helm-chart/nginx-ingress \
		--namespace=$(INGRESS_NAMESPACE) \
		--create-namespace

	@echo "============================================="
	@echo "2/5. Deploying Core Infrastructure & LiteLLM ($(SRC_NAMESPACE))..."
	@echo "============================================="
	helm upgrade --install postgresql ./helm-chart/postgresql \
		--namespace=$(SRC_NAMESPACE) \
		--set auth.username="$(POSTGRES_USER)" \
		--set auth.password="$(POSTGRES_PASSWORD)" \
		--set auth.database="$(POSTGRES_DB)" \
		--set primary.persistence.size=8Gi

	helm upgrade --install minio ./helm-chart/minio \
		--namespace=$(SRC_NAMESPACE) \
		--set mode=standalone \
		--set rootUser="$(MINIO_ROOT_USER)" \
		--set rootPassword="$(MINIO_ROOT_PASSWORD)" \
		--set 'buckets[0].name=vqa-images' \
		--set 'buckets[0].policy=public' \
		--set persistence.size=8Gi

	@helm repo add qdrant https://qdrant.github.io/qdrant-helm
	@helm dependency build ./helm-chart/qdrant
	helm upgrade --install qdrant ./helm-chart/qdrant \
		--namespace=$(SRC_NAMESPACE)

	helm upgrade --install litellm ./helm-chart/litellm \
		--namespace=$(SRC_NAMESPACE)

	@echo "============================================="
	@echo "3/5. Deploying VQA Application ($(SRC_NAMESPACE))..."
	@echo "============================================="
	helm upgrade --install backend ./helm-chart/backend \
		--namespace=$(SRC_NAMESPACE)

	helm upgrade --install frontend ./helm-chart/frontend \
		--namespace=$(SRC_NAMESPACE)

	@echo "============================================="
	@echo "4/5. Deploying Prometheus Monitoring Stack ($(MONITORING_NAMESPACE))..."
	@echo "============================================="
	@helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 
	@helm repo add grafana https://grafana.github.io/helm-charts 
	@helm dependency build ./helm-chart/monitoring/kube-prometheus-stack 
	helm upgrade --install monitoring ./helm-chart/monitoring/kube-prometheus-stack \
		--namespace=$(MONITORING_NAMESPACE) \
		--create-namespace \
		-f ./helm-chart/monitoring/kube-prometheus-stack.expanded.yaml

	@echo "============================================="
	@echo "5/5. Deploying ELK Logging Stack ($(LOGGING_NAMESPACE))..."
	@echo "============================================="
	-cd helm-chart/ELK && helmfile sync

	@echo "============================================="
	@echo "🎉 DEPLOYMENT COMPLETE!"
	@echo "============================================="
	@echo "Run 'make status' to check pod status across all namespaces."

status: ## View all Pods, Services, and Ingress across all namespaces
	@echo "============================================="
	@echo "📦 NAMESPACE: $(SRC_NAMESPACE)"
	@echo "============================================="
	@kubectl get pods,svc,ingress -n $(SRC_NAMESPACE)
	@echo ""
	@echo "============================================="
	@echo "📊 NAMESPACE: $(MONITORING_NAMESPACE)"
	@echo "============================================="
	@kubectl get pods,svc -n $(MONITORING_NAMESPACE)
	@echo ""
	@echo "============================================="
	@echo "🌐 NAMESPACE: $(INGRESS_NAMESPACE)"
	@echo "============================================="
	@kubectl get pods,svc -n $(INGRESS_NAMESPACE)
	@echo ""
	@echo "============================================="
	@echo "📑 NAMESPACE: $(LOGGING_NAMESPACE)"
	@echo "============================================="
	@kubectl get pods,svc -n $(LOGGING_NAMESPACE)

port-forward: ## Port-forward ALL internal services & dashboards (Grafana:3000, Kibana:5601, MinIO S3:9000, MinIO UI:9001, Qdrant:6333)
	@echo "Starting Port-Forward for all dashboards & APIs..."
	@echo "  -> Grafana:      http://localhost:3000 (Namespace: $(MONITORING_NAMESPACE))"
	@echo "  -> Kibana:       http://localhost:5601 (Namespace: $(LOGGING_NAMESPACE))"
	@echo "  -> MinIO S3 API: http://localhost:9000 (Namespace: $(SRC_NAMESPACE))"
	@echo "  -> MinIO Web UI: http://localhost:9001 (Namespace: $(SRC_NAMESPACE))"
	@echo "  -> Qdrant UI:    http://localhost:6333/dashboard (Namespace: $(SRC_NAMESPACE))"
	@echo "Press Ctrl+C to stop all port-forwarding."
	@trap 'kill 0' EXIT; \
	kubectl port-forward svc/monitoring-grafana 3000:80 -n $(MONITORING_NAMESPACE) & \
	kubectl port-forward svc/kibana 5601:5601 -n $(LOGGING_NAMESPACE) & \
	kubectl port-forward svc/minio 9000:9000 -n $(SRC_NAMESPACE) & \
	kubectl port-forward svc/minio-console 9001:9001 -n $(SRC_NAMESPACE) & \
	kubectl port-forward svc/qdrant 6333:6333 -n $(SRC_NAMESPACE) & \
	wait

destroy: ## Clean up old default namespace releases and uninstall all namespaces
	@echo "Cleaning up default namespace..."
	-helm uninstall frontend -n default
	-helm uninstall backend -n default
	-helm uninstall litellm -n default
	-helm uninstall minio -n default
	-helm uninstall postgresql -n default
	-helm uninstall monitoring -n default
	@echo "Uninstalling releases from separated namespaces..."
	-helm uninstall frontend -n $(SRC_NAMESPACE)
	-helm uninstall backend -n $(SRC_NAMESPACE)
	-helm uninstall litellm -n $(SRC_NAMESPACE)
	-helm uninstall minio -n $(SRC_NAMESPACE)
	-helm uninstall qdrant -n $(SRC_NAMESPACE)
	-helm uninstall postgresql -n $(SRC_NAMESPACE)
	-helm uninstall ingress-nginx -n $(INGRESS_NAMESPACE)
	-helm uninstall monitoring -n $(MONITORING_NAMESPACE)
	@echo "All releases uninstalled."
